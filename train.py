import argparse
import os
import json
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
import time

from algorithm import get_algorithm_class
from dataset import get_dataset
import utils


def evaluate(algorithm,loader,args):
    algorithm.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    for step,(data,target) in enumerate(loader):
        data,target = data.cuda(),target.cuda()
        with torch.no_grad():
            loss,acc = algorithm.predict(data,target)
            
            val_loss.update(loss,data.size(0))
            val_acc.update(acc,data.size(0))
    
    return val_loss.avg, val_acc.avg
    
def main(args):
    # Save config
    args.work_dir = os.path.join(f'./logs/{args.dataset}/{args.method}',f'{args.work_dir}')
    os.makedirs(args.work_dir,exist_ok=True)
    with open(os.path.join(args.work_dir,'config.json'),'w') as f:
        json.dump(args.__dict__,f,indent=2)
    
    # Set seed
    utils.init_distributed_mode(args)
    utils.set_seed(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # Dataset
    train_loader, val_loader, test_loader = get_dataset(args)
            
    # Algorithm
    args.lr = args.lr * args.batch_size / 512
    algorithm = get_algorithm_class(args.method)(args)
    lr_scheduler = utils.cosine_scheduler(
        base_value=args.lr,
        final_value=0,
        total_iters=args.step,
        warmup_iters=args.warmup_iters
    )
    # Algorithm specific process
    if args.method == 'CoTuning' or 'CoTuningStochCA':
        # Train relationship
        end = time.time()
        algorithm.get_relationship(train_loader)
        print(f"{time.time()-end} sec elapsed for relationship learning")
        
    # train & validation & test
    train_iterator = iter(train_loader)
    epoch = 0
    if args.world_size > 1:
        train_loader.sampler.set_epoch(epoch)
    algorithm.train()
    train_log = {'step_time':utils.AverageMeter()}
    end = time.time()
    for step in range(args.step):
        # Grab data
        try:
            data,target = next(train_iterator)
        except StopIteration:
            if args.world_size > 1:
                train_loader.sampler.set_epoch(epoch+1)
            train_iterator = iter(train_loader)
            data,target = next(train_iterator)
            epoch += 1
        
        param_groups = algorithm.optimizer.param_groups
        for i,param_group in enumerate(param_groups):
            param_group["lr"] = lr_scheduler[step]
        
        # Update
        data, target = data.cuda(non_blocking=True),target.cuda(non_blocking=True)
        step_log = algorithm.update(data,target)
            
        # iter log
        for k,v in step_log.items():
            if step == 0:
                train_log[k] = utils.AverageMeter()
            train_log[k].update(v,data.size(0))
        train_log['step_time'].update(time.time()-end)
        end = time.time()

        # Logging & Test
        if step % args.val_freq == 0 or step == args.step-1:
            result = {
                'train_step' : step,
            }
            for k,v in train_log.items():
                v.sync_multi_gpus(args.world_size)
                result[k] = v.avg
                v.reset()
            
            # Validation
            if args.use_val:
                val_loss, val_acc = evaluate(algorithm,val_loader,args)
            else:
                val_loss = 0
                val_acc = 0
            
            # Test
            test_loss, test_acc = evaluate(algorithm,test_loader,args)
            
            result['val_loss'] = val_loss
            result['val_acc'] = val_acc
            result['test_loss'] = test_loss
            result['test_acc'] = test_acc
            
            if step == 0:
                utils.print_row([k for k,v in result.items()],colwidth=12)
            utils.print_row([v for k,v in result.items()],colwidth=12)
            
            if args.rank == 0:
                with open(os.path.join(args.work_dir,'train_log.json'),'a') as f:
                    f.write(json.dumps(result,sort_keys=True)+"\n")
            
            algorithm.train()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer learning')
    parser.add_argument('--data_dir', default='/data/transfer_benchmarks',type=str,
                        help='Data directory')
    parser.add_argument('--dataset', default='CUB',type=str,
                        help='Data directory')
    parser.add_argument('--work_dir', default='./result',type=str,
                        help='Working directory')
    parser.add_argument('--batch_size', default=16,type=int,
                        help='Batch size')
    parser.add_argument('--optim', default='adamw',type=str,choices=['adam','adamw'],
                        help='Optimizer for adaptation')
    parser.add_argument('--lr', default=5e-04,type=float,
                        help='Initial Learning rate. Will be multiplied by (batch_size/512)')
    parser.add_argument('--cls_lr', default=1,type=float,
                        help='multiply LR for classification head')
    parser.add_argument('--wd', default=0.05,type=float,
                        help='Weight decay')
    parser.add_argument('--ratio', default=1.0,type=float,
                        help='Training sample ratio')
    parser.add_argument('--seed', default=0,type=int,
                        help='Random seed')
    parser.add_argument('--step', default=10000,type=int,
                        help='Training step')
    parser.add_argument('--warmup_iters', default=100,type=int,
                        help='LR warmup iterations')
    parser.add_argument('--val_freq', default=200,type=int,
                        help='Validation and test frequency')
    parser.add_argument('--input_size', default=448,type=int,
                        help='Input resolution')
    parser.add_argument('--method', default=None,type=str,
                        help='Transfer learning algorithm')
    parser.add_argument('--use_val', action="store_true",
                        help='Whether to use validation set for hyperparameter search')
    parser.add_argument('--local_rank', default=0,type=int,
                        help='Passed by launch')
    parser.add_argument("--dist_url", default="env://", type=str, 
                        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    
    # L2-SP
    parser.add_argument('--beta', default=0.01,type=float,
                        help='Control L2-SP reg term')
    
    # Co-tuning
    parser.add_argument('--ld', default=2.3,type=float,
                        help='Control Co-tuning loss weight')
    
    # BSS
    parser.add_argument('--num_singular', default=1,type=int,
                        help='Number of singular values to penalize')
    parser.add_argument('--eta', default=0.001,type=float,
                        help='Control BSS loss weight')
    
    # StochCA
    parser.add_argument('--ca_prob', default='lin_0.5_0.5',type=str,
                        help='Control Cross attention probability')
    
    args = parser.parse_args()
    main(args)