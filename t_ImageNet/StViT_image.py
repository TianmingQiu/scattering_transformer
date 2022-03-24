# ref: https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
# Path definition
from multiprocessing import reduction
import os
import sys
from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim import lr_scheduler
import time
import numpy as np
from kymatio.torch import Scattering2D
from einops import rearrange
from matplotlib import pyplot as plt

from models.vit_pytorch import ViT
from data_loading import data_loader

torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,5,7'
DEVICE_LIST = [0,1,2,3]

# DOWNLOAD_PATH = './input/dataset/Imagenet/Data/CLS-LOC'
DOWNLOAD_PATH = parent + '/input/dataset/tiny-imagenet-200'
SAVE_FOLDER = parent + '/checkpoint'
RESULT_FOLDER = parent + '/log'

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 1000

N_EPOCHS = 500
IMAGE_SIZE = 64
NUM_CLASS = 200
PATCH_SIZE = 2
DEPTH = 9
HEAD = 4
EMBED_DIM = 192
MLP_RATIO = 2
K = 25

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader,test_loader = data_loader(DOWNLOAD_PATH, image_size=IMAGE_SIZE, num_class=NUM_CLASS, batch_size_train=BATCH_SIZE_TRAIN, batch_size_test=BATCH_SIZE_TEST, workers=2, pin_memory=True)
scattering = Scattering2D(J=2, L=4, shape=(IMAGE_SIZE, IMAGE_SIZE), max_order=2)

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    quarter = int(len(data_loader)/4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        scattered_data, target = scattering(data).to(device), target.to(device)
        # scattered_data[:,:,0,:,:] /= 3
        scattered_data = rearrange(scattered_data, 'b c x h d -> b (c x) h d')
        optimizer.zero_grad()
        output = F.log_softmax(model(scattered_data), dim=1)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if i % quarter == 0:
            print('[' +  '{:5}'.format(i * len(scattered_data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            #plt.imshow(torch.permute(data[0], (1,2,0)))
            #plt.show()
            #scattered_data = scattering(data)
            #for i in range(scattered_data.shape[2]):
                #image = torch.permute(scattered_data,(0,2,3,4,1))[0,i]
                #plt.imshow(image / image.abs().mean() * 0.4)
               # print(image.abs().mean())
                #plt.show()
            #pause()
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history, acc_history):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = scattering(data).to(device), target.to(device)
            # data[:,:,0,:,:] /= 3
            data = rearrange(data, 'b c x h d -> b (c x) h d')
            output = F.log_softmax(model(data), dim=1)
            # loss = F.nll_loss(output, target, reduction='sum')
            loss = criterion(output, target)
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    acc = correct_samples / total_samples
    loss_history.append(avg_loss)
    acc_history.append(acc)

    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

start_time = time.time()
model = ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASS, channels=3*K,
        dim=EMBED_DIM, depth=DEPTH, heads=HEAD, mlp_dim=EMBED_DIM*MLP_RATIO, dropout=0.1, emb_dropout=0.1)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8)

model.to(device)
if device == 'cuda':
    #model = torch.nn.DataParallel(model) # make parallel
    cudnn.benchmark = True

train_loss_history, test_loss_history, accuracy_history = [], [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:' + '{:4}'.format(epoch), ' Learning rate: ' + '{:.1e}'.format(optimizer.param_groups[0]['lr']))
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history, accuracy_history)
    # scheduler.step(test_loss_history[-1])
    scheduler.step()

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)

save_path = SAVE_FOLDER + '/t_imagenet_d' + str(DEPTH)+'_h' + str(HEAD) + '_s.pth'
image_path = RESULT_FOLDER + '/t_imagenet_d' + str(DEPTH)+'_h' + str(HEAD) + '_s.png'

torch.save((model.state_dict(),accuracy_history,test_loss_history), save_path)
print('Model saved to', save_path)

# model,accuracy_history,test_loss_history = load(model,SAVE_FOLDER + '/t_imagenet_b50.pth')

plt.figure(figsize=(6,5))
plt.plot(np.arange(N_EPOCHS),torch.stack(accuracy_history).cpu().numpy(), c='black', label='ViT', linewidth=2)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('accuracy',fontsize=15)
plt.savefig(image_path, format='png', bbox_inches='tight')