import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim import lr_scheduler
import time
import os
from matplotlib import pyplot as plt

from models.vit_pytorch import ViT, ViT_scatter

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE_LIST = [0]

DOWNLOAD_PATH = './input/dataset'
SAVE_FOLDER = './checkpoint'
BATCH_SIZE_TRAIN = 20
BATCH_SIZE_TEST = 1000

transform_stl10 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4, 0.4, 0.4), (0.2, 0.2, 0.2)),
])

train_set = torchvision.datasets.STL10(DOWNLOAD_PATH, split='train', download=True,
                                       transform=transform_stl10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)

test_set = torchvision.datasets.STL10(DOWNLOAD_PATH, split='test', download=True,
                                      transform=transform_stl10)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time.time()
model = ViT(image_size=96, patch_size=8, num_classes=10, channels=3,
        dim=288, depth=10, heads=12, mlp_dim=288*4, dropout=0.1, emb_dropout=0.1)

model_s = ViT_scatter(image_size=96, patch_size=8, num_classes=10, channels=3,
        dim=288, depth=10, heads=12, mlp_dim=288*4, dropout=0.1, emb_dropout=0.1)

state_dict, _ = torch.load('checkpoint/0322/stl_vit.pth')
state_dict_s, _ = torch.load('checkpoint/0322/stl_vit_s.pth')
model.load_state_dict(state_dict)
model_s.load_state_dict(state_dict_s)
model.to(device)
model_s.to(device)

if device == 'cuda':
    cudnn.benchmark = True


# input_tensor = train_set[0][0].unsqueeze(0).cuda()
for data, target in train_loader:
    input_tensor = data.to('cuda')
    break
# model(input_tensor)
model_s(input_tensor)

def arr_mean(arrs):
    arr_add = lambda x,y:[a+b for (a,b) in zip(x,y)]
    res = [0 for _ in range(len(arrs[0]))]
    for arr in arrs:
        res = arr_add(arr,res)
    return [x / len(arrs) for x in res]

att_s = torch.load('attention_s.pth')
index = [0,1,8,9]
for i in index:
    plt.plot(arr_mean(att_s[i*20:i*20+20]),marker='o')
plt.ylim([35,60])
plt.xlabel('Sorted Attention Head')
plt.ylabel('Mean Distance')
plt.legend(['encoder_block0','encoder_block1','encoder_block8','encoder_block9'])
plt.grid()
plt.title('SViT-Patch to STL10')
plt.show()

# att = torch.load('attention.pth')
# plt.plot(list(zip(*att[-10:])))
# plt.legend(range(1,11))
# plt.show()

# att_s = torch.load('attention_s.pth')
# plt.plot(list(zip(*att_s[-10:])))
# plt.legend(range(1,11))
# plt.show()