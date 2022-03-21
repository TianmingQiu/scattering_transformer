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

from models.vit_pytorch import ViT
from models.t2t_vit import T2T_ViT 

torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
DEVICE_LIST = [0]

DOWNLOAD_PATH = './input/dataset'
SAVE_FOLDER = './checkpoint'
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_TEST = 1000

transform_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_cifar10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)

test_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_cifar10)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)

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

        if i % 20 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            target = target.to(device)
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

N_EPOCHS = 200

start_time = time.time()
# model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,
#             dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
model = T2T_ViT(img_size=32, num_classes=10, in_chans=3,
        embed_dim=512, depth=6, num_heads=8)
# model.load_state_dict(torch.load(SAVE_FOLDER + '/cifar_d2_b' + str(N_EPOCHS) + '.pth'))

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

save_path = SAVE_FOLDER + '/cifar_d6_b' + str(N_EPOCHS) + '.pth'
torch.save(model.state_dict(), save_path)
print('Model saved to', save_path)