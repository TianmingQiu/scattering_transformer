import os
from os import path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision
from scipy.io import loadmat
import PIL.Image

class Flowers102Dataset(Dataset):
    def __init__(self, dir, split = 'train', transform=None, target_transform=None):
        self.dir = path.join(dir, 'flowers102')
        self.transform = transform
        self.target_transform = target_transform
        self.img_folder = path.join(self.dir, 'jpg')
        self.label_path = path.join(self.dir,'imagelabels.mat')
        self.anno_path = path.join(self.dir,'setid.mat')

        self.img_labels = loadmat(self.label_path)['labels'][0]
        if split == 'train':
            self.image_idxs = loadmat(self.anno_path)['trnid'][0]
        elif split == 'test':
            self.image_idxs = loadmat(self.anno_path)['tstid'][0]
        elif split == 'validation':
            self.image_idxs = loadmat(self.anno_path)['valid'][0]
        else:
            raise Exception('Split not known to Flower102 dataset')
        

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        idx = self.image_idxs[idx]
        img_path = os.path.join(self.img_folder, 'image_' + str(idx).zfill(5) + '.jpg')
        image = PIL.Image.open(img_path).convert("RGB")
        label = self.img_labels[idx-1].astype('int64') - 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class tinyImageNet(Dataset):
    def __init__(self, dir, split = 'train', transform=None, target_transform=None, num_class = 10):
        self.dir = path.join(dir, 'tiny-imagenet-200')
        self.transform = transform
        self.target_transform = target_transform

        assert split in ['train','test']
        if split == 'train':
            self.dir = os.path.join(self.dir,'train')
            image_set = torchvision.datasets.ImageFolder(self.dir,transform)
        elif split == 'test':
            self.dir = os.path.join(self.dir,'val')
            image_set = torchvision.datasets.ImageFolder(self.dir,transform)

        indices = (torch.tensor(image_set.targets)[...,None]==torch.arange(num_class)).any(-1).nonzero(as_tuple=True)[0]
        self.data = torch.utils.data.Subset(image_set,indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ImageNet(Dataset):
    def __init__(self, dir, split = 'train', transform=None, target_transform=None, num_class = 10):
        self.dir = path.join(dir, 'Imagenet/Data/CLS-LOC')
        self.transform = transform
        self.target_transform = target_transform

        assert split in ['train','test']
        if split == 'train':
            self.dir = os.path.join(self.dir,'train')
            image_set = torchvision.datasets.ImageFolder(self.dir,transform)
        elif split == 'test':
            self.dir = os.path.join(self.dir,'val')
            image_set = torchvision.datasets.ImageFolder(self.dir,transform)

        indices = (torch.tensor(image_set.targets)[...,None]==torch.arange(num_class)).any(-1).nonzero(as_tuple=True)[0]
        self.data = torch.utils.data.Subset(image_set,indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EuroSAT(Dataset):
    def __init__(self, dir, split = 'train', transform=None, target_transform=None, num_class = 10, train_percentage = 70):
        self.dir = path.join(dir, 'EuroSAT/2750')
        self.transform = transform
        self.target_transform = target_transform

        # if train_percentage == None:
        #     train_percentage = 70
        assert 1 <= train_percentage <= 99, 'Train percentage must remain between 0 and 100'
        print('Train percentage set to {}%'.format(train_percentage))
        image_set = torchvision.datasets.ImageFolder(self.dir,transform)
        train_indices = [i for i in range(27000) if i % 100 < train_percentage]
        test_indices  = [i for i in range(27000) if i % 100 >=train_percentage]

        assert split in ['train','test']
        if split == 'train':
            self.data = torch.utils.data.Subset(image_set,train_indices)
        elif split == 'test':
            self.data = torch.utils.data.Subset(image_set,test_indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
