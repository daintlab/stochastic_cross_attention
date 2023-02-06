import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vision_transformer
from collections import OrderedDict
from copy import deepcopy
from torch.utils.data import DataLoader

import utils
from dataset import get_transform
# for stochCA
from vit_crossattention_v8 import vit_b_16 as student_vit
from vit_return_qkv import vit_b_16 as teacher_vit

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class ERM(nn.Module):
    def __init__(self,args):
        super(ERM,self).__init__()
        self.args = args
        
        # Define model
        if args.input_size != 224:
            # interpolate positional embedding
            pretrained_weight = torchvision.models.vit_b_16(pretrained=True).state_dict()
            new_dict = vision_transformer.interpolate_embeddings(
                image_size=args.input_size,
                patch_size=16,
                model_state=pretrained_weight
                )
            self.model = torchvision.models.vit_b_16(image_size=args.input_size)
            self.model.load_state_dict(new_dict)

        else:
            self.model = torchvision.models.vit_b_16(pretrained=True)

        del self.model.heads
        self.model.heads = nn.Identity()
        self.classifier = nn.Linear(768,args.num_classes)
        self.network = nn.Sequential(
            OrderedDict([
                ('feat',self.model),
                ('classifier',self.classifier),
            ])
        ).cuda()
        if args.world_size > 1:
            self.network = nn.parallel.DistributedDataParallel(
                self.network,device_ids=[args.gpu]
            )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [
                {'params' : self.network.feat.parameters(), 'lr' : args.lr},
                {'params' : self.network.classifier.parameters(), 'lr' : args.lr * args.cls_lr},
            ],
            weight_decay=args.wd
        )
        
    def update(self,x,y):
        output = self.network(x)
        loss = self.criterion(output,y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc,_= utils.accuracy(output,y,topk=(1,5))
        return {'train_loss':loss.item(), 'train_acc':acc[0].item()}
    
    def predict(self,x,y):
        # output = self.classifier(self.model(x))
        output = self.network(x)
        loss = self.criterion(output,y)
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return loss.item(), acc[0].item()
        
class L2SP(nn.Module):
    '''
    refer https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/delta.py#L35
    '''
    def __init__(self,args):
        super(L2SP,self).__init__()
        self.args = args
        
        # Define model
        if args.input_size != 224:
            # interpolate positional embedding
            pretrained_weight = torchvision.models.vit_b_16(pretrained=True).state_dict()
            new_dict = vision_transformer.interpolate_embeddings(
                image_size=args.input_size,
                patch_size=16,
                model_state=pretrained_weight
                )
            self.model = torchvision.models.vit_b_16(image_size=args.input_size)
            self.model.load_state_dict(new_dict)

        else:
            self.model = torchvision.models.vit_b_16(pretrained=True)

        del self.model.heads
        self.model.heads = nn.Identity()
        self.classifier = nn.Linear(768,args.num_classes)
        self.network = nn.Sequential(
            OrderedDict([
                ('feat',self.model),
                ('classifier',self.classifier),
            ])
        ).cuda()
        if args.world_size > 1:
            self.network = nn.parallel.DistributedDataParallel(
                self.network,device_ids=[args.gpu]
            )
        
        # for L2-SP regularization
        self.beta = args.beta
        self.source_weight = {}
        for name,param in self.model.named_parameters():
            self.source_weight[name] = deepcopy(param).detach().cuda(args.gpu)
                
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [
                {'params' : self.model.parameters(), 'weight_decay' : 0},
                {'params' : self.classifier.parameters(), 'weight_decay' : args.wd},
            ],
            lr=args.lr,
        )
    
    def l2_sp_reg(self,model):
        output = 0.0
        for name,param in model.named_parameters():
            output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output
        
    def update(self,x,y):
        # output = self.classifier(self.model(x))
        output = self.network(x)
        
        train_loss = self.criterion(output,y)
        reg_loss = self.l2_sp_reg(self.model)
        loss = train_loss + self.beta * reg_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return {'train_loss':loss.item(), 'L2-SP':reg_loss.item(), 'train_acc':acc[0].item()}
    
    def predict(self,x,y):
        # output = self.classifier(self.model(x))
        output = self.network(x)
        loss = self.criterion(output,y)
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return loss.item(), acc[0].item()
    
class BSS(nn.Module):
    '''
    refer : https://github.com/thuml/Batch-Spectral-Shrinkage
    refer : https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/bss.py
    Only support single-GPU
    '''
    def __init__(self,args):
        super(BSS,self).__init__()
        self.args = args
        # Define model
        if args.input_size != 224:
            # interpolate positional embedding
            pretrained_weight = torchvision.models.vit_b_16(pretrained=True).state_dict()
            new_dict = vision_transformer.interpolate_embeddings(
                image_size=args.input_size,
                patch_size=16,
                model_state=pretrained_weight
                )
            self.backbone = torchvision.models.vit_b_16(image_size=args.input_size)
            self.backbone.load_state_dict(new_dict)
        else:
            self.backbone = torchvision.models.vit_b_16(pretrained=True)

        del self.backbone.heads
        self.backbone.heads = nn.Identity()
        self.backbone.cuda()
        self.classifier = nn.Linear(768,args.num_classes).cuda()
        
        # Define loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        modules = nn.ModuleList([self.backbone,self.classifier])
        self.optimizer = torch.optim.AdamW(
            modules.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    
    def bss_loss(self,feature):
        bss = 0
        u,s,v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.args.num_singular):
            bss += torch.pow(s[num-1-i],2)
        return bss

    def update(self,x,y):
        feat = self.backbone(x)
        output = self.classifier(feat)
        
        loss_base = self.criterion(output,y)
        loss_bss = self.bss_loss(feat)
        loss = loss_base + self.args.eta * loss_bss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc,_= utils.accuracy(output,y,topk=(1,5))
        return {
            'train_loss':loss.item(),
            'bss_loss' : loss_bss.item(),
            'train_acc':acc[0].item()
        }
    
    def predict(self,x,y):
        output = self.classifier(self.backbone(x))
        loss = self.criterion(output,y)
        acc,_= utils.accuracy(output,y,topk=(1,5))
        return loss.item(), acc[0].item()
        
    
class CoTuning(nn.Module):
    '''
    refer : https://github.com/thuml/CoTuning
    Only support single-GPU
    '''
    def __init__(self,args):
        super(CoTuning,self).__init__()
        self.args = args
        
        # Define model
        if args.input_size != 224:
            # interpolate positional embedding
            pretrained_weight = torchvision.models.vit_b_16(pretrained=True).state_dict()
            new_dict = vision_transformer.interpolate_embeddings(
                image_size=args.input_size,
                patch_size=16,
                model_state=pretrained_weight
                )
            self.backbone = torchvision.models.vit_b_16(image_size=args.input_size)
            self.backbone.load_state_dict(new_dict)
        else:
            self.backbone = torchvision.models.vit_b_16(pretrained=True)

        self.classifier_s = deepcopy(self.backbone.heads[0]).cuda()
        del self.backbone.heads
        self.backbone.heads = nn.Identity()
        
        self.classifier_t = nn.Linear(self.classifier_s.in_features,args.num_classes).cuda()
        self.backbone.cuda()
        
        # Define loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {'params' : self.backbone.parameters(), 'lr' : args.lr},
                {'params' : self.classifier_s.parameters(), 'lr' : args.lr},
                {'params' : self.classifier_t.parameters(), 'lr' : args.lr * 10},
            ],
            weight_decay=args.wd
        )
    
    def get_relationship(self,train_loader):
        print("Computing relationship")
        # Split train data -> train/val to learn relationship
        val_ratio = 0.1
        if self.args.ratio < 1: # Train loader contains Subset
            org_trainset = deepcopy(train_loader.dataset.dataset)
            indices = deepcopy(train_loader.dataset.indices)
        else: # Train loader contains ImageFolder
            org_trainset = deepcopy(train_loader.dataset)
            indices = list(range(len(org_trainset)))
        
        np.random.shuffle(indices)
        train_indices = indices[int(len(indices)*val_ratio):]
        val_indices = indices[:int(len(indices)*val_ratio)]            
        train_dataset = torch.utils.data.Subset(org_trainset,train_indices)
        val_dataset = torch.utils.data.Subset(org_trainset,val_indices)

        train_dataset.dataset.transform = get_transform(mode='val')
        val_dataset.dataset.transform = get_transform(mode='val')
        
        determin_train_loader = DataLoader(
            train_dataset,batch_size=self.args.batch_size,shuffle=False,pin_memory=True
        )
        val_loader = DataLoader(
            train_dataset,batch_size=self.args.batch_size,shuffle=False,pin_memory=True
        )
        
        # Learn relationship
        self.backbone.eval()
        self.classifier_s.eval()
        self.classifier_t.eval()
        
        def get_feature(loader):
            with torch.no_grad():
                target_label_list = []
                source_label_list = []
                for i,(data,target) in enumerate(loader):
                    target_label_list.append(target)
                    data,target = data.cuda(),target.cuda()
                    output_s = self.classifier_s(self.backbone(data))
                    
                    output_s = output_s.detach().cpu().numpy()
                    source_label_list.append(output_s)
            all_source_labels = np.concatenate(source_label_list,0)
            all_target_labels = np.concatenate(target_label_list,0)
            return all_source_labels, all_target_labels

        train_source_labels, train_target_labels = get_feature(determin_train_loader)
        val_source_labels, val_target_labels = get_feature(val_loader)
        relationship = utils.relationship_learning(
            train_source_labels, train_target_labels,val_source_labels,val_target_labels
        )
        
        self.relationship = relationship
    
    def cotuning_loss(self, output, target):
        loss = - target * F.log_softmax(output,dim=-1)
        loss = torch.mean(torch.sum(loss,dim=-1))
        return loss
    
    def update(self,x,y):
        # Get source label through relationship
        y_s = torch.from_numpy(self.relationship[y.cpu()]).cuda().float()
        
        # forward
        feat = self.backbone(x)
        output_s = self.classifier_s(feat)
        output_t = self.classifier_t(feat)
        
        loss_t = self.criterion(output_t,y)
        loss_s = self.cotuning_loss(output_s,y_s)
        loss = loss_t + self.args.ld * loss_s
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc,_= utils.accuracy(output_t,y,topk=(1,5))
        
        return {
            'train_loss':loss.item(), 
            'cotuning_loss':loss_s.item(),
            'train_acc':acc[0].item()
        }
        
    def predict(self,x,y):
        output = self.classifier_t(self.backbone(x))
        loss = self.criterion(output,y)
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return loss.item(), acc[0].item()
        
class StochCA(nn.Module):
    def __init__(self,args):
        super(StochCA,self).__init__()
        self.args = args
        ca_prob = self.ca_prob_handler()
        
        if args.input_size != 224:
        # interpolate positional embedding
            pretrained_weight = student_vit(pretrained=True,ca_prob=ca_prob).state_dict()
            new_dict = vision_transformer.interpolate_embeddings(
                image_size=args.input_size,
                patch_size=16,
                model_state=pretrained_weight
                )
            self.student = student_vit(image_size=args.input_size,ca_prob=ca_prob)
            self.teacher = teacher_vit(image_size=args.input_size)
            self.student.load_state_dict(new_dict)
            self.teacher.load_state_dict(new_dict)

        else:
            self.student = student_vit(image_size=args.input_size,ca_prob=ca_prob)
            self.teacher = teacher_vit(image_size=args.input_size)
        
        del self.student.heads
        self.student.heads = nn.Identity()
        
        del self.teacher.heads
        self.teacher.heads = nn.Identity()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.cuda()
        self.student.cuda()
        self.classifier = nn.Linear(768,args.num_classes).cuda()

        if args.world_size > 1:
            self.student = nn.parallel.DistributedDataParallel(
                self.student,device_ids=[args.gpu]
            )
            self.classifier = nn.parallel.DistributedDataParallel(
                self.classifier,device_ids=[args.gpu]
            )
        
        self.criterion = nn.CrossEntropyLoss()
        # modules = nn.ModuleList([self.student,self.classifier])
        # self.optimizer = torch.optim.AdamW(
        #     modules.parameters(),
        #     lr=args.lr,
        #     weight_decay=args.wd
        # )
        self.optimizer = torch.optim.AdamW(
            [
                {'params' : self.student.parameters(), 'lr' : args.lr},
                {'params' : self.classifier.parameters(), 'lr' : args.lr * args.cls_lr},
            ],
            weight_decay=args.wd
        )
        
    def ca_prob_handler(self):
        '''
        assume hparams['ca_prob'] to be 'linear_<start>_<end>' formation
        # TODO : More various ca_prob scehdule
        '''
        assert len(self.args.ca_prob.split('_')) == 3
        _,start,end = self.args.ca_prob.split('_')
        ca_prob = torch.linspace(float(start),float(end),12).tolist()
        return ca_prob

    def update(self,x,y):
        # Get teacher's q,k,v
        with torch.no_grad():
            self.teacher.eval()
            _,query,key,val = self.teacher(x)
        
        # Forward with teachers k,v
        output = self.classifier(self.student(x,key=key,val=val))
        loss = self.criterion(output,y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return {'train_loss':loss.item(), 'train_acc':acc[0].item()}
    
    def predict(self,x,y):
        output = self.classifier(self.student(x))
        loss = self.criterion(output,y)
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return loss.item(), acc[0].item()
        
        
class CoTuningStochCA(nn.Module):
    def __init__(self,args):
        super(CoTuningStochCA,self).__init__()
        self.args = args
        ca_prob = self.ca_prob_handler()
        
        if args.input_size != 224:
        # interpolate positional embedding
            pretrained_weight = student_vit(pretrained=True,ca_prob=ca_prob).state_dict()
            new_dict = vision_transformer.interpolate_embeddings(
                image_size=args.input_size,
                patch_size=16,
                model_state=pretrained_weight
                )
            self.student = student_vit(image_size=args.input_size,ca_prob=ca_prob)
            self.teacher = teacher_vit(image_size=args.input_size)
            self.student.load_state_dict(new_dict)
            self.teacher.load_state_dict(new_dict)

        else:
            self.student = student_vit(image_size=args.input_size,ca_prob=ca_prob)
            self.teacher = teacher_vit(image_size=args.input_size)
        self.classifier_s = deepcopy(self.teacher.heads[0]).cuda()
        del self.student.heads
        self.student.heads = nn.Identity()
        
        del self.teacher.heads
        self.teacher.heads = nn.Identity()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.cuda()
        self.student.cuda()
        self.classifier_t = nn.Linear(768,args.num_classes).cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        modules = nn.ModuleList([self.student,self.classifier_s,self.classifier_t])
        self.optimizer = torch.optim.AdamW(
            modules.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
        
    def get_relationship(self,train_loader):
        print("Computing relationship")
        # Split train data -> train/val to learn relationship
        val_ratio = 0.1
        if self.args.ratio < 1: # Train loader contains Subset
            org_trainset = deepcopy(train_loader.dataset.dataset)
            indices = deepcopy(train_loader.dataset.indices)
        else: # Train loader contains ImageFolder
            org_trainset = deepcopy(train_loader.dataset)
            indices = list(range(len(org_trainset)))
        
        np.random.shuffle(indices)
        train_indices = indices[int(len(indices)*val_ratio):]
        val_indices = indices[:int(len(indices)*val_ratio)]            
        train_dataset = torch.utils.data.Subset(org_trainset,train_indices)
        val_dataset = torch.utils.data.Subset(org_trainset,val_indices)

        train_dataset.dataset.transform = get_transform(mode='val')
        val_dataset.dataset.transform = get_transform(mode='val')
        
        determin_train_loader = DataLoader(
            train_dataset,batch_size=self.args.batch_size,shuffle=False,pin_memory=True
        )
        val_loader = DataLoader(
            train_dataset,batch_size=self.args.batch_size,shuffle=False,pin_memory=True
        )
        
        # Learn relationship
        self.teacher.eval()
        self.classifier_s.eval()
        self.classifier_t.eval()
        
        def get_feature(loader):
            with torch.no_grad():
                target_label_list = []
                source_label_list = []
                for i,(data,target) in enumerate(loader):
                    target_label_list.append(target)
                    data,target = data.cuda(),target.cuda()
                    feat, _,_,_ = self.teacher(data)
                    output_s = self.classifier_s(feat)
                    
                    output_s = output_s.detach().cpu().numpy()
                    source_label_list.append(output_s)
            all_source_labels = np.concatenate(source_label_list,0)
            all_target_labels = np.concatenate(target_label_list,0)
            return all_source_labels, all_target_labels

        train_source_labels, train_target_labels = get_feature(determin_train_loader)
        val_source_labels, val_target_labels = get_feature(val_loader)
        relationship = utils.relationship_learning(
            train_source_labels, train_target_labels,val_source_labels,val_target_labels
        )
        
        self.relationship = relationship
        
    def ca_prob_handler(self):
        '''
        assume hparams['ca_prob'] to be 'linear_<start>_<end>' formation
        # TODO : More various ca_prob scehdule
        '''
        assert len(self.args.ca_prob.split('_')) == 3
        _,start,end = self.args.ca_prob.split('_')
        ca_prob = torch.linspace(float(start),float(end),12).tolist()
        return ca_prob
    
    def cotuning_loss(self, output, target):
        loss = - target * F.log_softmax(output,dim=-1)
        loss = torch.mean(torch.sum(loss,dim=-1))
        return loss

    def update(self,x,y):
        # Get source label through relationship
        y_s = torch.from_numpy(self.relationship[y.cpu()]).cuda().float()
        
        # Get teacher's q,k,v
        with torch.no_grad():
            self.teacher.eval()
            _,query,key,val = self.teacher(x)
        
        # Forward with teachers k,v
        feat = self.student(x,key=key,val=val)
        output_s = self.classifier_s(feat)
        output_t = self.classifier_t(feat)
        
        loss_t = self.criterion(output_t,y)
        loss_s = self.cotuning_loss(output_s,y_s)
        loss = loss_t + self.args.ld * loss_s
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc,_= utils.accuracy(output_t,y,topk=(1,5))
        
        return {'train_loss':loss.item(), 'cotuning_loss':loss_s.item(),'train_acc':acc[0].item()}
    
    def predict(self,x,y):
        output = self.classifier_t(self.student(x))
        loss = self.criterion(output,y)
        acc,_= utils.accuracy(output,y,topk=(1,5))
        
        return loss.item(), acc[0].item()
        