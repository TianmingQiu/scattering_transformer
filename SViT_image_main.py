
# from torch.utils.data import DataLoader

# from input.data import *
# from src import *

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim import lr_scheduler

import time
import os
import yaml
import argparse

from kymatio.torch import Scattering2D

from einops import rearrange
from matplotlib import pyplot as plt

from src.models.vit_pytorch import ViT
from src.models.vit_no_cls_token import ViT_no_cls_token


def main():
    # argparser --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Scattering ViT')
    parser.add_argument('--config', default='config.yaml')
    args, _ = parser.parse_known_args()

    # Load config.yaml whether for 'inference' or 'train'
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)  # config is a dict

    # First GPU:ID will be used as main GPU - first entry in config.cuda_devices
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'] == 'gpu':
        device = torch.device('cuda:{}'.format(config['cuda_devices'][0]))

        # Check for config error and missmatch with accessible torch devices
        if torch.cuda.device_count() < len(config['cuda_devices']):
            raise Exception('Sorry, received cuda_devices (specified in config.yaml) exceeds available GPUs')

    # random seed
    if config['manual_seed']:
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])

    # dataset ----------------------------------------------------------------------------------------------------------
    dataset_dir = './input/dataset'
    if config['dataset'][0] == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root=dataset_dir, train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=dataset_dir, train=False,
                                             download=True, transform=transform)

    elif config['dataset'][0] == 'CIFAR10':
        transform = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomResizedCrop(32),
            # transforms.Resize(32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                               download=True, transform=transform)


    
    else:
        raise Exception('No defined dataset')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                              shuffle=True, num_workers=config['num_workers'])

    test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                             shuffle=False, num_workers=config['num_workers'])
    # evaluator = TestEval(testloader, device, config)


    # model --------------------------------------------------------------------------------------------------------
    scattering = Scattering2D(
        J=config['scattering_J'], 
        L=config['scattering_L'], 
        shape=(config['scattering_shape'], config['scattering_shape'])
        )

    start_time = time.time()

    # model = ViT(image_size=16, patch_size=2, num_classes=10, channels=9*3,
    #         dim=192, depth=9, heads=6, mlp_dim=256, dropout=0.1, emb_dropout=0.1)
    model = ViT_no_cls_token(image_size=16, patch_size=2, num_classes=10, channels=51,
            dim=192, depth=3, heads=6, mlp_dim=256, dropout=0.1, emb_dropout=0.1)
    model.to(device)
    # if device == 'cuda':
    if len(config['cuda_devices']) > 1:
        model = torch.nn.DataParallel(model, device_ids=config['cuda_devices']) # make parallel
        cudnn.benchmark = True

    # optimizer ----------------------------------------------------------------------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

    train_loss_history, test_loss_history = [], []
    for epoch in range(config['epoch']):
        print('Epoch:' + '{:4}'.format(epoch+1), ' Learning rate: ' + '{:.1e}'.format(optimizer.param_groups[0]['lr']))
        total_samples = len(train_loader.dataset)
        model.train()

        for i, (data, target) in enumerate(train_loader):
            scattered_data, target = scattering(data).to(device), target.to(device)
            # print(scattered_data.shape)
            # scattered_data[:,:,0,:,:] /= 3
            scattered_data = rearrange(scattered_data, 'b c x h d -> b (c x) h d')
            optimizer.zero_grad()
            output = F.log_softmax(model(scattered_data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % config['eval_freq'] == 0:
                print('[' +  '{:5}'.format(i * len(scattered_data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i / len(train_loader)) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                train_loss_history.append(loss.item())
                evaluate(model, scattering, test_loader, test_loss_history, device)
        # scheduler.step(loss)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

    if not os.path.exists(config['save_folder']):
        os.mkdir(config['save_folder'])

    save_path = config['save_folder'] + '/cifar_d6_b' + str(config['epoch']) + '_s.pth'
    torch.save(model.state_dict(), save_path)
    print('Model saved to', save_path)


# def train_epoch(model, scattering, optimizer, data_loader, loss_history, device):

#             plt.imshow(torch.permute(data[0], (1,2,0)))
#             plt.show()
#             scattered_data = scattering(data)
#             for i in range(scattered_data.shape[2]):
#                 image = torch.permute(scattered_data,(0,2,3,4,1))[0,i]
#                 plt.imshow(image / image.abs().mean() * 0.4)
#                 print(image.abs().mean())
#                 plt.show()
#             loss_history.append(loss.item())

def evaluate(model, scattering, data_loader, loss_history, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = scattering(data).to(device), target.to(device)
            # data[:,:,0,:,:] /= 3
            data = rearrange(data, 'b c x h d -> b (c x) h d')
            output = F.log_softmax(model(data), dim=1)
            _, pred = torch.max(output, dim=1)
            
            total += target.size(0)
            correct += (pred == target).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
    main()
