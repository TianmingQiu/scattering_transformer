import torch
import torchvision
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from kymatio.torch import Scattering2D

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

DOWNLOAD_PATH = './input/dataset'
# SAVE_FOLDER = '../checkpoint'
# BATCH_SIZE_TRAIN = 1
# BATCH_SIZE_TEST = 1



def cosine_similarity(x,y, head=None):
    x = x.reshape(-1) - x.mean()
    y = y.reshape(-1) - y.mean()
    if head:
        x = x[head[0]:head[1]]
        y = y[head[0]:head[1]]
    return sum(x * y) / torch.norm(x) / torch.norm(y)

def scatter_full_img(image, plot_feat=True):
    scatter2d_full = Scattering2D(J=1,L=4,shape=(96,96))

    # Inspect full image
    scatter_full = scatter2d_full(image.contiguous()).permute(1,2,3,0)
    if plot_feat:
        for i in range(scatter_full.shape[0]):
            # Set large scale for high pass images for better visuals
            img_scale = 10 if i > 0 else 1
            plt.imshow(scatter_full[i] * img_scale)
            plt.axis('off')
            plt.show()
    return scatter_full


def scatter_patch(image, patch_size, p_coord, plot_feat=True):
    scatter2d = Scattering2D(J=1,L=4,shape=(patch_size, patch_size))
    # Inspect image patch
    p_x, p_y = p_coord[0], p_coord[1]
    patch1 = image[:, p_x:p_x+patch_size, p_y:p_y+patch_size]
    tensor2imgshow(patch1)
    scatter_patch = scatter2d(patch1.contiguous()).permute(1,2,3,0)
    if plot_feat:
        for i in range(scatter_patch.shape[0]):
            # Set large scale for high pass images for better visuals
            img_scale = 10 if i > 0 else 1
            plt.imshow(scatter_patch[i] * img_scale)
            plt.axis('off')
            plt.show()
    return scatter_patch

def tensor2imgshow(img_tensor):
    plt.imshow(img_tensor.permute(1,2,0).detach().cpu())
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    transform_stl10 = transforms.Compose([
    # transforms.Grayscale(),
    transforms.ToTensor(),
    
    ])

    train_set = torchvision.datasets.STL10(DOWNLOAD_PATH, split='train', download=False,
                                        transform=transform_stl10)
    # train_set = torchvision.datasets.CIFAR10(DOWNLOAD_PATH, train=True, download=False,
    #                                     transform=transform_stl10)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, pin_memory=True)
    # for i, (data, target) in enumerate(train_loader):
    image = train_set[1][0]
    
    tensor2imgshow(image)
    scatter_full_img(image, plot_feat=True)
    p1 = scatter_patch(image, 32, (80, 40), )
    p2 = scatter_patch(image, 32, (50, 30), )
    p3 = scatter_patch(image, 32, (0, 0), )
    print(
        cosine_similarity(p1, p2),
        cosine_similarity(p1, p3),
        cosine_similarity(p2, p3),
        cosine_similarity(torch.tensor([0.0,0.0, 1.0]), torch.tensor([1.0,0.0, 0.0]))
    )

    


