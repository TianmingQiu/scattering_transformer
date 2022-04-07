from torchvision import transforms
import torchvision
import torch
from .custom_dataset import *

transform_mnist = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

transform_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_stl10 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4, 0.4, 0.4), (0.2, 0.2, 0.2)),
])

transform_flowers = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4, 0.4, 0.4), (0.2, 0.2, 0.2)),
])

transform_tImageNet = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229,0.224,0.225])
])

transform_ImageNet = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229,0.224,0.225])
])

transform_FashionMNIST = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.225])
    # transforms.Normalize(mean=[0.2859], std=[0.3530])
    # transforms.Normalize(mean=[0.1307], std=[0.3081])
])


def generate_dataset_information(DATASET_TYPE,DOWNLOAD_PATH,BATCH_SIZE_TRAIN,BATCH_SIZE_TEST,
                                patch_size=None,depth=None,head=None,embed_dim=None):
    DATASET_TYPE = DATASET_TYPE.upper()
    print('Reading dataset information: {}'.format(DATASET_TYPE))
    if DATASET_TYPE == 'STL10':
        train_set = torchvision.datasets.STL10(DOWNLOAD_PATH, split='train', download=True,
                                        transform=transform_stl10)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = torchvision.datasets.STL10(DOWNLOAD_PATH, split='test', download=True,
                                        transform=transform_stl10)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 96
        PATCH_SIZE = 8
        NUM_CLASS = 10
        DEPTH = 10
        HEAD = 8
        EMBED_DIM = 192
        CHANNELS = 3
        
    elif DATASET_TYPE == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=True, download=True,
                                        transform=transform_cifar10)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=False, download=True,
                                        transform=transform_cifar10)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 32
        PATCH_SIZE = 4
        NUM_CLASS = 10
        DEPTH = 10
        HEAD = 8
        EMBED_DIM = 192
        CHANNELS = 3

    elif DATASET_TYPE == 'FLOWERS':
        train_set = Flowers102Dataset(DOWNLOAD_PATH, split='test', transform=transform_flowers)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = Flowers102Dataset(DOWNLOAD_PATH, split='train',  transform=transform_flowers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 96
        PATCH_SIZE = 8
        NUM_CLASS = 102
        DEPTH = 10
        HEAD = 8
        EMBED_DIM = 192
        CHANNELS = 3

    elif DATASET_TYPE == 'TINYIMAGENET':
        NUM_CLASS = 50
        train_set = tinyImageNet(DOWNLOAD_PATH, split='train', transform=transform_tImageNet, num_class=NUM_CLASS)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = tinyImageNet(DOWNLOAD_PATH, split='test',  transform=transform_tImageNet, num_class=NUM_CLASS)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 64
        PATCH_SIZE = 8
        NUM_CLASS = 50
        DEPTH = 9
        HEAD = 4
        EMBED_DIM = 192
        CHANNELS = 3

    elif DATASET_TYPE == 'IMAGENET':
        NUM_CLASS = 10
        train_set = ImageNet(DOWNLOAD_PATH, split='train', transform=transform_ImageNet, num_class=NUM_CLASS)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = ImageNet(DOWNLOAD_PATH, split='test',  transform=transform_ImageNet, num_class=NUM_CLASS)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 96
        PATCH_SIZE = 8
        DEPTH = 9
        HEAD = 4
        EMBED_DIM = 192
        CHANNELS = 3

    elif DATASET_TYPE == 'FASHIONMNIST':
        NUM_CLASS = 10
        train_set = torchvision.datasets.FashionMNIST(DOWNLOAD_PATH, train=True, transform=transform_FashionMNIST)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = torchvision.datasets.FashionMNIST(DOWNLOAD_PATH, train=True,  transform=transform_FashionMNIST)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 28
        PATCH_SIZE = 4
        DEPTH = 1
        HEAD = 4
        EMBED_DIM = 48
        CHANNELS = 1

    elif DATASET_TYPE == 'EUROSAT':
        NUM_CLASS = 10
        train_set = EuroSAT(DOWNLOAD_PATH, split='train', transform=transform_stl10)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
        test_set = EuroSAT(DOWNLOAD_PATH, split='test',  transform=transform_stl10)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)
        IMAGE_SIZE = 64
        PATCH_SIZE = 8
        DEPTH = 9
        HEAD = 8
        EMBED_DIM = 192
        CHANNELS = 3

    else:
        raise Exception('Dataset {} not supported.'.format(DATASET_TYPE))

    # Override
    if patch_size:
        PATCH_SIZE=patch_size
    if depth:
        DEPTH=depth
    if head:
        HEAD=head
    if embed_dim:
        EMBED_DIM=embed_dim
    
    print('Dataset reading complete')

    return train_loader,test_loader,IMAGE_SIZE,PATCH_SIZE,NUM_CLASS,DEPTH,HEAD,EMBED_DIM,CHANNELS
