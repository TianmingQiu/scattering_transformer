import matplotlib.pyplot as plt
import torch
import torchvision

_,s  = torch.load('checkpoint/stl_vit.pth')
_,ss = torch.load('checkpoint/stl_vit_s.pth')
_,p  = torch.load('checkpoint/stl_psvit.pth')

plt.plot(s)
plt.plot(ss)
plt.plot(p)
plt.legend(['Vit','Scatter Vit','PS ViT'])
plt.show()
