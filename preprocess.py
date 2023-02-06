import os
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from scipy import io
import pandas as pd
import shutil

'''
Generate train/validation/test split
Use 10% of train split as validation split(with random seed=1234)
<train>,<val>,<test> directories will be generated under <data_dir>/<dataset>

<train_all> will contain original train split data under <data_dir>/<dataset>
if there is no official train_validation split

Available dataset : Aircraft, CUB, Car, Dog
'''
dataset = 'Aircraft'
data_dir = '/nas/datahub/transfer_benchmarks'

### Inspect ####
data_path = os.path.join(data_dir,dataset)
for split in ['train_all','train','val','test']:
    current_dir = os.path.join(data_path,split)
    classes = os.listdir(current_dir)
    total = 0
    for class_ in classes:
        num_data = len(os.listdir(os.path.join(current_dir,class_)))
        # print(f"{dataset} {split} {class_} : {num_data}")
        total += num_data
    
    print(f"\n {dataset} {split} : {total} \n")
import ipdb;ipdb.set_trace()

if dataset == 'CUB':
    train_id,train_class = [],[]
    path = os.path.join(data_dir,dataset,'CUB_200_2011')
    img_txt = open(os.path.join(path,'images.txt')).readlines()
    split_txt = open(os.path.join(path,'train_test_split.txt')).readlines()
    for img_line,split_line in zip(img_txt,split_txt):
        id_, name = img_line.strip().split(' ')
        id2, split = split_line.strip().split(' ')
        class_ = int(name[:3])-1
        assert id_ == id2
        if split == '1':
            train_id.append(id_)
            train_class.append(class_)
    
    dest_dir = os.path.join(data_dir,dataset)
    for img_line in img_txt:
        id_, name = img_line.strip().split(' ')
        if id_ in train_id:
            split = 'train_all'
            file_path = os.path.join(path,'images',name)
            dest_path = os.path.join(dest_dir,split)
            os.makedirs(os.path.join(dest_path,name.split('/')[0]),exist_ok=True)
            dest_path = os.path.join(dest_path,name)
            shutil.copy(file_path,dest_path)
    
    train_id,val_id = train_test_split(train_id,test_size=0.1,random_state=1234,stratify=train_class)
    for img_line in img_txt:
        id_, name = img_line.strip().split(' ')
        dest_dir = os.path.join(data_dir,dataset)
        os.makedirs(os.path.join(dest_dir,'train'),exist_ok=True)
        os.makedirs(os.path.join(dest_dir,'val'),exist_ok=True)
        os.makedirs(os.path.join(dest_dir,'test'),exist_ok=True)
        if id_ in train_id:
            split = 'train'
        elif id_ in val_id:
            split = 'val'
        else:
            split = 'test'
        file_path = os.path.join(path,'images',name)
        
        dest_path = os.path.join(dest_dir,split)
        os.makedirs(os.path.join(dest_path,name.split('/')[0]),exist_ok=True)
        dest_path = os.path.join(dest_path,name)
        shutil.copy(file_path,dest_path)
    

elif dataset == 'Car':
    meta = io.loadmat(os.path.join(data_dir, dataset, 'cars_annos.mat'))['annotations'][0]
    meta = pd.DataFrame(meta)
    
    # train id, train class
    train_ids = meta[meta['test'] == 0]['relative_im_path'].values
    train_class = meta[meta['test'] == 0]['class'].values

    train_ids = [f.item() for f in train_ids]
    train_class = [f.item()-1 for f in train_class]
    train_id,val_id = train_test_split(train_ids,test_size=0.1,random_state=1234,stratify=train_class)
    
    for index,row in meta.iterrows():
        file_path = row['relative_im_path'].item()
        class_ = row['class'].item()
        file_name = file_path.split('/')[-1]
        
        dest_dir = os.path.join(data_dir,dataset)
        os.makedirs(os.path.join(dest_dir,'train_all'),exist_ok=True)
        
        if file_path in train_ids:
            dest_path = os.path.join(dest_dir,'train_all')
            os.makedirs(os.path.join(dest_path,f"class{class_}"),exist_ok=True)
            dest_path = os.path.join(dest_path,f"class{class_}/{file_name}")
            file_path = os.path.join(data_dir,dataset,file_path)
            shutil.copy(file_path,dest_path)

    for index, row in meta.iterrows():
        file_path = row['relative_im_path'].item()
        class_ = row['class'].item()
        file_name = file_path.split('/')[-1]

        dest_dir = os.path.join(data_dir,dataset)
        os.makedirs(os.path.join(dest_dir,'train'),exist_ok=True)
        os.makedirs(os.path.join(dest_dir,'val'),exist_ok=True)
        os.makedirs(os.path.join(dest_dir,'test'),exist_ok=True)

        if file_path in train_id:
            split = 'train'
        elif file_path in val_id:
            split = 'val'
        else:
            split = 'test'
        
        dest_path = os.path.join(dest_dir,split)
        os.makedirs(os.path.join(dest_path,f"class{class_}"),exist_ok=True)
        dest_path = os.path.join(dest_path,f"class{class_}/{file_name}")
        file_path = os.path.join(data_dir,dataset,file_path)
        shutil.copy(file_path,dest_path)

elif dataset == 'Aircraft':
    meta_path = os.path.join(data_dir,dataset,'fgvc-aircraft-2013b/data')
    train_txt = open(os.path.join(meta_path,'images_variant_train.txt')).readlines()
    val_txt = open(os.path.join(meta_path,'images_variant_val.txt')).readlines()
    test_txt = open(os.path.join(meta_path,'images_variant_test.txt')).readlines()
    
    train_all_txt = open(os.path.join(meta_path,'images_variant_trainval.txt')).readlines()

    for line in train_all_txt:
        id_ = line.strip().split(' ')[0]
        class_ = ' '.join(line.strip().split(' ')[1:])
        file_path = os.path.join(meta_path,'images',f'{id_}.jpg')
        dest_path = os.path.join(data_dir,dataset,f'train_all/{class_}')
        os.makedirs(dest_path,exist_ok=True)
        dest_path = os.path.join(dest_path,f'{id_}.jpg')
        shutil.copy(file_path,dest_path)
        
    for line in train_txt:
        id_ = line.strip().split(' ')[0]
        class_ = ' '.join(line.strip().split(' ')[1:])
        file_path = os.path.join(meta_path,'images',f'{id_}.jpg')
        dest_path = os.path.join(data_dir,dataset,f'train/{class_}')
        os.makedirs(dest_path,exist_ok=True)
        dest_path = os.path.join(dest_path,f'{id_}.jpg')
        shutil.copy(file_path,dest_path)

    for line in val_txt:
        id_ = line.strip().split(' ')[0]
        class_ = ' '.join(line.strip().split(' ')[1:])
        file_path = os.path.join(meta_path,'images',f'{id_}.jpg')
        dest_path = os.path.join(data_dir,dataset,f'val/{class_}')
        os.makedirs(dest_path,exist_ok=True)
        dest_path = os.path.join(dest_path,f'{id_}.jpg')
        shutil.copy(file_path,dest_path)

    for line in test_txt:
        id_ = line.strip().split(' ')[0]
        class_ = ' '.join(line.strip().split(' ')[1:])
        file_path = os.path.join(meta_path,'images',f'{id_}.jpg')
        dest_path = os.path.join(data_dir,dataset,f'test/{class_}')
        os.makedirs(dest_path,exist_ok=True)
        dest_path = os.path.join(dest_path,f'{id_}.jpg')
        shutil.copy(file_path,dest_path)

elif dataset == 'Dog':
    data_path = os.path.join(data_dir,dataset)
    train_meta = io.loadmat(os.path.join(data_path, 'train_list.mat'))
    test_meta = io.loadmat(os.path.join(data_path, 'test_list.mat'))
    
    train_all_id = [f.item() for f in train_meta['file_list'].flatten()]
    train_all_class = [f.item()-1 for f in train_meta['labels'].flatten()]
    test_id = [f.item() for f in test_meta['file_list'].flatten()]
    
    train_id, val_id = train_test_split(train_all_id,test_size=0.1,random_state=1234,stratify=train_all_class)
    
    for id_ in train_all_id:
        # copy to train_all
        file_path = os.path.join(data_path,'Images',id_)
        class_name,file_name = id_.split('/')
        dest_path = os.path.join(data_path,f'train_all/{class_name}')
        os.makedirs(dest_path,exist_ok=True)
        dest_path = os.path.join(dest_path,file_name)
        #shutil.copy(file_path,dest_path)
        
        # copy to train or val
        split = 'train' if id_ in train_id else 'val'
        dest_path = os.path.join(data_path,f'{split}/{class_name}')
        os.makedirs(dest_path,exist_ok=True)
        dest_path = os.path.join(dest_path,file_name)
        shutil.copy(file_path,dest_path)
        
    # for id_ in test_id:
    #     # copy to test
    #     file_path = os.path.join(data_path,'Images',id_)
    #     class_name,file_name = id_.split('/')
    #     dest_path = os.path.join(data_path,f'test/{class_name}')
    #     os.makedirs(dest_path,exist_ok=True)
    #     dest_path = os.path.join(dest_path,file_name)
    #     shutil.copy(file_path,dest_path)

