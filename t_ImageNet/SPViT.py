# Path definition
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

# Standard imports
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# From the repository
from models.vit_pytorch import ViT, ViT_scatter, scatter_patch_ViT
from data_loading import data_loader

# DOWNLOAD_PATH = './input/dataset/Imagenet/Data/CLS-LOC'
DOWNLOAD_PATH = parent + '/input/dataset/tiny-imagenet-200'
SAVE_FOLDER = parent + '/checkpoint'
RESULT_FOLDER = parent + '/log'

torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
DEVICE_LIST = [0,1,2,3]

# Hyperparameters
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
N_EPOCHS = 200

IMAGE_SIZE = 64
SCATTER_LAYER = 3
SCATTER_ANGLE = 4
NUM_CLASS = 200
PATCH_SIZE = 8
DEPTH = 9
HEAD = 4
EMBED_DIM = 3*((IMAGE_SIZE/(2**SCATTER_LAYER))**2)*(1+SCATTER_ANGLE)
EMBED_DIM = int(EMBED_DIM)
MLP_RATIO = 2

train_loader,test_loader = data_loader(DOWNLOAD_PATH, image_size=IMAGE_SIZE, num_class=NUM_CLASS, batch_size_train=BATCH_SIZE_TRAIN, batch_size_test=BATCH_SIZE_TEST, workers=2, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history, acc_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    acc_history.append(correct_samples / total_samples)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

start_time = time.time()

model = scatter_patch_ViT(image_size=IMAGE_SIZE, scatter_layer = SCATTER_LAYER, scatter_angle = SCATTER_ANGLE,  patch_size = 8, num_classes=NUM_CLASS, channels=3,
        dim=EMBED_DIM, depth=DEPTH, heads=HEAD, mlp_dim=EMBED_DIM*MLP_RATIO, dropout=0.1, emb_dropout=0.1)
# model.load_state_dict(torch.load(SAVE_FOLDER + '/cifar_d2_b' + str(N_EPOCHS) + '.pth'))
# model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

model.to(device)
if device == 'cuda':
    # model = torch.nn.DataParallel(model) # make parallel
    cudnn.benchmark = True

train_loss_history, test_loss_history, accuracy_history = [], [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:' + '{:4}'.format(epoch), ' Learning rate: ' + '{:.1e}'.format(optimizer.param_groups[0]['lr']))
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history, accuracy_history)
    scheduler.step(test_loss_history[-1])

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)
    
save_path = SAVE_FOLDER + '/t_imagenet_stvit_' + str(PATCH_SIZE)+ '-' + str(DEPTH) + '.pth'
image_path = RESULT_FOLDER + '/t_imagenet_stvit_' + str(PATCH_SIZE)+ '-' + str(DEPTH) + '.png'

torch.save((model.state_dict(),accuracy_history,test_loss_history), save_path)
print('Model saved to', save_path)

# model,accuracy_history,test_loss_history=torch.load(SAVE_FOLDER + '/imagenet_b50.pth')
plt.figure(figsize=(6,5))
plt.plot(np.arange(N_EPOCHS),torch.stack(accuracy_history).cpu().numpy(), c='black', label='ViT', linewidth=2)
plt.xlabel('epoch',fontsize=15)
plt.xlabel('accuracy',fontsize=15)
plt.savefig(image_path, format='png', bbox_inches='tight')