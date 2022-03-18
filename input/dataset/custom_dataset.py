import os
from os import path
from torch.utils.data import Dataset
from torchvision.io import read_image
from scipy.io import loadmat
import PIL.Image

class Flowers102Dataset(Dataset):
    def __init__(self, dir, split = 'train', transform=None, target_transform=None):
        self.dir = path.join(dir, 'flowers102')
        self.transform = transform
        self.target_transform = target_transform
        self.img_folder = path.join(self.dir, 'jpg')
        self.label_path = path.join(self.dir,'imagelabels.mat')
        self.anno_path = path.join(self.dir,'setid.mat')

        self.img_labels = loadmat(self.label_path)['labels'][0]
        if split == 'train':
            self.image_idxs = loadmat(self.anno_path)['trnid'][0]
        elif split == 'test':
            self.image_idxs = loadmat(self.anno_path)['tstid'][0]
        elif split == 'validation':
            self.image_idxs = loadmat(self.anno_path)['valid'][0]
        else:
            raise Exception('Split not known to Flower102 dataset')


    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        idx = self.image_idxs[idx]
        img_path = os.path.join(self.img_folder, 'image_' + str(idx).zfill(5) + '.jpg')
        image = PIL.Image.open(img_path).convert("RGB")
        label = self.img_labels[idx-1].astype('int64') - 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
