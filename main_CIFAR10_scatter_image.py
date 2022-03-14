# ref: https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim import lr_scheduler
import time
import os
from kymatio.torch import Scattering2D
from einops import rearrange
from matplotlib import pyplot as plt

from vit_pytorch import ViT

torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
DEVICE_LIST = [0,1,2,3]

DOWNLOAD_PATH = './input/dataset'
SAVE_FOLDER = './checkpoint'
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_TEST = 1000

transform_mnist = torchvision.transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

transform_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_cifar10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)

test_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_cifar10)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scattering = Scattering2D(J=1, L=4, shape=(32, 32))

def pause():
    input('Paused')
    print('Resumed')


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        scattered_data, target = scattering(data).to(device), target.to(device)
        # scattered_data[:,:,0,:,:] /= 3
        scattered_data = rearrange(scattered_data, 'b c x h d -> b (c x) h d')
        optimizer.zero_grad()
        output = F.log_softmax(model(scattered_data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print('[' +  '{:5}'.format(i * len(scattered_data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            plt.imshow(torch.permute(data[0], (1,2,0)))
            plt.show()
            scattered_data = scattering(data)
            for i in range(scattered_data.shape[2]):
                image = torch.permute(scattered_data,(0,2,3,4,1))[0,i]
                plt.imshow(image / image.abs().mean() * 0.4)
                print(image.abs().mean())
                plt.show()
            pause()
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history):
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
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

N_EPOCHS = 25

start_time = time.time()
# model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,
#             dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
model = ViT(image_size=16, patch_size=2, num_classes=10, channels=3 * 5,
        dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model) # make parallel
    cudnn.benchmark = True

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:' + '{:4}'.format(epoch), ' Learning rate: ' + '{:.1e}'.format(optimizer.param_groups[0]['lr']))
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history)
    scheduler.step(test_loss_history[-1])

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

save_path = SAVE_FOLDER + '/cifar_d6_b' + str(N_EPOCHS) + '_s.pth'
torch.save(model.state_dict(), save_path)
print('Model saved to', save_path)