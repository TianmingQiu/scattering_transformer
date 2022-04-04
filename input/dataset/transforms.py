from torchvision import transforms

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
    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229,0.224,0.225])
])