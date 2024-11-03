from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Any
import os


class MyDataset(Dataset):
    def __init__(self, dir, transform=None, target_transform=None, loader=None):
        self.main_dir = dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.targets = []
        self.instances = self.make_instances()
        self.loader = loader

        if loader is None:
            self.loader = lambda x: Image.open(x).convert('RGB')

    def make_instances(self):
        instances = []
        targets = []
        for target_class in sorted(self.class_to_idx.keys()):
                class_index = self.class_to_idx[target_class]
                target_dir = os.path.join(self.main_dir, target_class)
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = path, class_index
                        targets.append(class_index)
                        instances.append(item)
        self.targets = torch.tensor(targets)
        return instances
    
    def __getitem__(self,index:int) -> Tuple[Any,Any]:
        path, target = self.instances[index]
        instance = self.loader(path)
        if self.transform is not None:
            instance = self.transform(instance)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return instance,target
    
    def __len__(self) -> int:
        return len(self.instances)

def build_dataset(args):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])
    # If cats dataset
    if "cats" in args.data_path: 
        return get_test_val_train_splits(MyDataset(args.data_path, transform), args.batch_size)
    # If ImageNet A dataset
    else:
        test_imgs_path = args.data_path + "/test"
        train_imgs_path = args.data_path + "/train" 
        ds_train = MyDataset(train_imgs_path, transform)
        ds_test = MyDataset(test_imgs_path, transform)
        return get_test_val_train_splits_ima(ds_train, ds_test, args.batch_size)


def get_test_val_train_splits(ds, bs):
    train_idx, temp_idx = train_test_split(np.arange(len(ds)),test_size=0.3,shuffle=True,stratify=ds.targets, random_state=42)
    valid_idx, test_idx = train_test_split(temp_idx,test_size=0.5,shuffle=True,stratify=ds.targets[temp_idx], random_state=42)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    test_sampler  = torch.utils.data.SubsetRandomSampler(test_idx)

    dl_train = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=train_sampler)
    dl_valid = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=valid_sampler)
    dl_test  = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=test_sampler)
    return dl_train, dl_valid, dl_test

def get_test_val_train_splits_ima(ds_train, ds_test, bs):
    train_idx = np.arange(len(ds_train))
    np.random.seed(42)
    np.random.shuffle(train_idx)
    test_idx = np.arange(len(ds_test))
    np.random.seed(42)
    np.random.shuffle(test_idx)
    #valid_idx, test_idx = train_test_split(np.arange(len(ds_test)),test_size=0.5,shuffle=True,stratify=ds_test.targets, random_state=42)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
   

    dl_train = torch.utils.data.DataLoader(ds_train,batch_size=bs,sampler=train_sampler)
    dl_test = torch.utils.data.DataLoader(ds_test,batch_size=bs,sampler=test_sampler)

    return dl_train, dl_test, dl_test
    
#def main():
#    print("Testing dataset")
#    dir_path = "/home/cassio/git/CAS-ViT/carros"
#    dataset = build_dataset(dir_path)
#    print(dataset.instances)


#if __name__ == "__main__":
#    main()

