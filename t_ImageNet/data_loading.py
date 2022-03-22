import os
import torch
import torchvision
from torchvision import transforms

def normalize_transform(image_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229,0.224,0.225])
    ])
    return transform

def augmented_transform(image_size=64):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandAugment(num_ops=3,magnitude=9,num_magnitude_bins=31,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229,0.224,0.225])
    ])
    return transform
    
def train_dataset(data_dir,image_size=64):
    train_dir = os.path.join(data_dir,'train')
    train_transforms = augmented_transform(image_size)
    train_set = torchvision.datasets.ImageFolder(train_dir,train_transforms)
    return train_set

def test_dataset(data_dir,image_size=64):
    test_dir = os.path.join(data_dir, 'val')
    test_transforms = augmented_transform(image_size)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)
    return test_dataset

def data_loader(data_dir,image_size=64, num_class=10, batch_size_train=100, batch_size_test=1000, workers=2, pin_memory=True):
    train_set = train_dataset(data_dir,image_size)
    test_set = test_dataset(data_dir,image_size)
    train_indices = (torch.tensor(train_set.targets)[...,None]==torch.arange(num_class)).any(-1).nonzero(as_tuple=True)[0]
    train_data = torch.utils.data.Subset(train_set,train_indices)
    test_indices = (torch.tensor(test_set.targets)[...,None]==torch.arange(num_class)).any(-1).nonzero(as_tuple=True)[0]
    test_data = torch.utils.data.Subset(test_set,test_indices)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, test_loader