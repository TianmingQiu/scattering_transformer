from random import shuffle
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim import lr_scheduler
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from models.vit_pytorch import ViT, ViT_scatter

torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
DEVICE_LIST = [0]

DOWNLOAD_PATH = './input/dataset/Imagenet/Data/CLS-LOC'
SAVE_FOLDER = './checkpoint'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000
IMAGE_SIZE = 224
NUM_CLASS = 1000

def normalize_transform():
    return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229,0.224,0.225])

def train_dataset(data_dir):
    train_dir = os.path.join(data_dir,'train')
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])
    train_set = torchvision.datasets.ImageFolder(train_dir,train_transforms)
    return train_set

def test_dataset(data_dir):
    test_dir = os.path.join(data_dir, 'val')
    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize_transform()
    ])
    test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)
    return test_dataset

def data_loader(data_dir, num_class=10, batch_size_train=100, batch_size_test=1000, workers=2, pin_memory=True):
    train_set = train_dataset(data_dir)
    test_set = test_dataset(data_dir)
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

train_loader,test_loader = data_loader(DOWNLOAD_PATH, num_class=NUM_CLASS, batch_size_train=BATCH_SIZE_TRAIN, batch_size_test=BATCH_SIZE_TEST, workers=2, pin_memory=True)

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

        if i % 10 == 0:
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

N_EPOCHS = 200

start_time = time.time()
# model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,
#             dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
model = ViT(image_size=IMAGE_SIZE, patch_size=16, num_classes=NUM_CLASS, channels=3,
        dim=200, depth=6, heads=8, mlp_dim=200*4, dropout=0.1, emb_dropout=0.1)
# model.load_state_dict(torch.load(SAVE_FOLDER + '/cifar_d2_b' + str(N_EPOCHS) + '.pth'))
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

save_path = SAVE_FOLDER + '/imagenet_b' + str(N_EPOCHS) + '.pth'
torch.save((model.state_dict(),accuracy_history,test_loss_history), save_path)
print('Model saved to', save_path)

# model,accuracy_history,test_loss_history=torch.load(SAVE_FOLDER + '/imagenet_b50.pth')
plt.figure(figsize=(6,5))
plt.plot(np.arange(N_EPOCHS),torch.stack(accuracy_history).cpu().numpy(), c='black', label='ViT', linewidth=2)
plt.xlabel('epoch',fontsize=15)
plt.xlabel('accuracy',fontsize=15)
plt.savefig(SAVE_FOLDER + 'imagenet_b200.png', format='png', bbox_inches='tight')