import torch
import torchvision
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from kymatio.torch import Scattering2D

torch.manual_seed(42)
torch.cuda.manual_seed(42)

DOWNLOAD_PATH = './input/dataset'
SAVE_FOLDER = './checkpoint'
BATCH_SIZE_TRAIN = 1
BATCH_SIZE_TEST = 1

transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.STL10(DOWNLOAD_PATH, split='train', download=True,
                                       transform=transform_stl10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)

def cosine_similarity(x,y):
    x = x.reshape(-1) - x.mean()
    y = y.reshape(-1) - y.mean()
    return sum(x * y) / torch.norm(x) / torch.norm(y)

for i, (data, target) in enumerate(train_loader):
    image = data[0].cpu()

    scatter2d = Scattering2D(J=1,L=4,shape=(8,8))
    scatter2d_full = Scattering2D(J=1,L=4,shape=(96,96))

    # Inspect full image
    scatter_full = scatter2d_full(image.contiguous()).permute(1,2,3,0)
    for i in range(scatter_full.shape[0]):
        # Set large scale for high pass images for better visuals
        img_scale = 10 if i > 0 else 1
        plt.imshow(scatter_full[i] * img_scale)
        plt.show()


    # Inspect image patch
    patch1 = image[:,24:32,24:32]
    scatter_patch = scatter2d(patch1.contiguous()).permute(1,2,3,0)
    for i in range(scatter_patch.shape[0]):
        # Set large scale for high pass images for better visuals
        img_scale = 10 if i > 0 else 1
        plt.imshow(scatter_patch[i] * img_scale)
        plt.show()

    if i > -1:
        break