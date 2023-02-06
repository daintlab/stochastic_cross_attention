import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_transform(mode='train'):
    if mode == 'train':
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(384, scale=(0.7, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
            transforms.Resize((550,550)),
            transforms.RandomCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((550,550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    return transform

def get_dataset(args):
    data_path = os.path.join(args.data_dir,args.dataset)
    
    if args.use_val:
        train_dataset = ImageFolder(data_path+'/train',transform=get_transform('train'))
        val_dataset = ImageFolder(data_path+'/val',transform=get_transform('val'))
    else:
        train_dataset = ImageFolder(data_path+'/train_all',transform=get_transform('train'))
        val_dataset = None
    test_dataset = ImageFolder(data_path+'/test',transform=get_transform('val'))
    args.num_classes = len(test_dataset.classes)
    
    if args.ratio < 1.0:    
        train_labels = torch.tensor(train_dataset.targets)
        class_, counts = torch.unique(train_labels,return_counts=True)
        indices = []
        for i,count in enumerate(counts):
            idx = torch.where(train_labels==class_[i])[0]
            idx = idx[torch.randperm(len(idx))[:int(len(idx)*args.ratio)]]
            indices.extend(idx.tolist())
        train_dataset = torch.utils.data.Subset(train_dataset,indices)

    if args.world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset,shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=int(args.batch_size/args.world_size),
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=16,pin_memory=True)
        
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=16,pin_memory=True) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=16,pin_memory=True)
    
    return train_loader, val_loader, test_loader
