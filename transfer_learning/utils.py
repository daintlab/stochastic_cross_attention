import numpy as np
import torch
import random
from datetime import datetime
import os
import torch.nn as nn
import torch.distributed as dist

from sklearn.linear_model import LogisticRegression

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        return
        # os.environ['MASTER_ADDR'] = '127.0.0.1'
        # os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync_multi_gpus(self,world_size):
        if not world_size > 1:
            return
        t = torch.tensor([self.count,self.sum],dtype=torch.float64,device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.avg = self.sum / self.count

        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.6f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)
    
    
def cosine_scheduler(base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule



## For Co-tuning ###
# Copy pasted from https://github.com/thuml/CoTuning/blob/main/module/relationship_learning.py

def calibrate(logits, labels):
    """
    calibrate by minimizing negative log likelihood.
    :param logits: pytorch tensor with shape of [N, N_c]
    :param labels: pytorch tensor of labels
    :return: float
    """
    scale = nn.Parameter(torch.ones(
        1, 1, dtype=torch.float32), requires_grad=True)
    optim = torch.optim.LBFGS([scale])

    def loss():
        optim.zero_grad()
        lo = nn.CrossEntropyLoss()(logits * scale, labels)
        lo.backward()
        return lo

    state = optim.state[scale]
    for i in range(20):
        optim.step(loss)
        print(f'calibrating, {scale.item()}')
        if state['n_iter'] < optim.state_dict()['param_groups'][0]['max_iter']:
            break

    return scale.item()


def softmax_np(x):
    max_el = np.max(x, axis=1, keepdims=True)
    x = x - max_el
    x = np.exp(x)
    s = np.sum(x, axis=1, keepdims=True)
    return x / s


def relationship_learning(train_logits, train_labels, validation_logits, validation_labels):
    """
    :param train_logits (ImageNet logits): [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param train_labels:  [N], where 0 <= each number < N_t, and N_t is the number of target dataset
    :param validation_logits (ImageNet logits): [N, N_p]
    :param validation_labels:  [N]
    :return: [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """

    # convert logits to probabilities
    train_probabilities = softmax_np(train_logits * 0.8840456604957581)
    validation_probabilities = softmax_np(
        validation_logits * 0.8840456604957581)

    all_probabilities = np.concatenate(
        (train_probabilities, validation_probabilities))
    all_labels = np.concatenate((train_labels, validation_labels))

    Cs = []
    accs = []
    classifiers = []
    for C in [1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        cls_ = LogisticRegression(
            multi_class='multinomial', C=C, fit_intercept=False)
        cls_.fit(train_probabilities, train_labels)
        val_predict = cls_.predict(validation_probabilities)
        val_acc = np.sum((val_predict == validation_labels).astype(
            np.float)) / len(validation_labels)
        Cs.append(C)
        accs.append(val_acc)
        classifiers.append(cls_)

    accs = np.asarray(accs)
    ind = int(np.argmax(accs))
    cls_ = classifiers[ind]
    del classifiers

    validation_logits = np.matmul(validation_probabilities, cls_.coef_.T)
    validation_logits = torch.from_numpy(validation_logits.astype(np.float32))
    validation_labels = torch.from_numpy(validation_labels)

    scale = calibrate(validation_logits, validation_labels)

    p_target_given_pretrain = softmax_np(
        cls_.coef_.T * scale)  # shape of [N_p, N_c], conditional probability p(target_class | pre-trained class)

    # in the paper, both ys marginal and yt marginal are computed
    # here we only use ys marginal to make sure p_pretrain_given_target is a valid conditional probability
    # (make sure p_pretrain_given_target[i] sums up to 1)
    pretrain_marginal = np.mean(all_probabilities, axis=0).reshape(
        (-1, 1))  # shape of [N_p, 1]
    p_joint_distribution = (p_target_given_pretrain * pretrain_marginal).T
    p_pretrain_given_target = p_joint_distribution / \
        np.sum(p_joint_distribution, axis=1, keepdims=True)

    return p_pretrain_given_target

