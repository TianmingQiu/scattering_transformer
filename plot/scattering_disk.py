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

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from kymatio import Scattering2D
from PIL import Image
import torchvision
from torchvision import transforms

IMAGE_PATH = parent + '/input/dataset'
transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    ])

train_set = torchvision.datasets.STL10(IMAGE_PATH, split='train', download=False,
                                        transform=transform_stl10)
# image size 3*96*96
src = train_set[0][0]
if src.shape[0] == 3:
    src = torchvision.transforms.functional.rgb_to_grayscale(src)

trans = torchvision.transforms.ToPILImage()
src_img = trans(src)



L = 4
J = 3
scattering = Scattering2D(J=J, shape=(src.shape[1],src.shape[2]), L=L, max_order=2, frontend='torch')
scat_coeffs = scattering(src)
print("coeffs shape: ", scat_coeffs.shape)
# Invert colors
scat_coeffs= -scat_coeffs

len_order_1 = J*L
scat_coeffs_order_1 = scat_coeffs[:,1:1+len_order_1, :, :]
norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap="gray")
# Mapper of coefficient amplitude to a grayscale color for visualisation.

len_order_2 = (J*(J-1)//2)*(L**2)
scat_coeffs_order_2 = scat_coeffs[:,1+len_order_1:, :, :]
norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap="gray")
# Mapper of coefficient amplitude to a grayscale color for visualisation.

# Retrieve spatial size
window_rows, window_columns = scat_coeffs.shape[2:]
print("nb of (order 1, order 2) coefficients: ", (len_order_1, len_order_2))

# Define figure size and grid on which to plot input digit image, first-order and second-order scattering coefficients
fig = plt.figure(figsize=(47, 15))
spec = fig.add_gridspec(ncols=3, nrows=1)

gs = gridspec.GridSpec(1, 3, wspace=0.1)
gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])
gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[2])

# Start by plotting input digit image and invert colors
ax = plt.subplot(gs[0])
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(src_img, cmap='gray', interpolation='nearest', aspect='auto')

# Plot first-order scattering coefficients
# for i in range(src_tensor.shape[0]):

ax = plt.subplot(gs[1])
ax.set_xticks([])
ax.set_yticks([])

l_offset = int(L - L / 2 - 1)  # follow same ordering as Kymatio for angles

for row in range(window_rows):
    for column in range(window_columns):
        ax = fig.add_subplot(gs_order_1[row, column], projection='polar')
        ax.axis('off')
        coefficients = scat_coeffs_order_1[:,:, row, column]
        for j in range(J):
            for l in range(L):
                coeff = coefficients[:,l + j * L]
                color = mapper_order_1.to_rgba(coeff)
                angle = (l_offset - l) * np.pi / L
                radius = 2 ** (-j - 1)
                ax.bar(x=angle,
                       height=radius,
                       width=np.pi / L,
                       bottom=radius,
                       color=color)
                ax.bar(x=angle + np.pi,
                       height=radius,
                       width=np.pi / L,
                       bottom=radius,
                       color=color)

# Plot second-order scattering coefficients
ax = plt.subplot(gs[2])
ax.set_xticks([])
ax.set_yticks([])

for row in range(window_rows):
    for column in range(window_columns):
        ax = fig.add_subplot(gs_order_2[row, column], projection='polar')
        ax.axis('off')
        coefficients = scat_coeffs_order_2[:,:, row, column]
        for j1 in range(J - 1):
            for j2 in range(j1 + 1, J):
                for l1 in range(L):
                    for l2 in range(L):
                        coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                      (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                        # indexing a bit complex which follows the order used by Kymatio to compute
                        # scattering coefficients
                        coeff = coefficients[:,coeff_index]
                        color = mapper_order_2.to_rgba(coeff)
                        # split along angles first-order quadrants in L quadrants, using same ordering
                        # as Kymatio (clockwise) and center (with the 0.5 offset)
                        angle = (l_offset - l1) * np.pi / L + (L // 2 - l2 - 0.5) * np.pi / (L ** 2)
                        radius = 2 ** (-j1 - 1)
                        # equal split along radius is performed through height variable
                        ax.bar(x=angle,
                               height=radius / 2 ** (J - 2 - j1),
                               width=np.pi / L ** 2,
                               bottom=radius + (radius / 2 ** (J - 2 - j1)) * (J - j2 - 1),
                               color=color)
                        ax.bar(x=angle + np.pi,
                               height=radius / 2 ** (J - 2 - j1),
                               width=np.pi / L ** 2,
                               bottom=radius + (radius / 2 ** (J - 2 - j1)) * (J - j2 - 1),
                               color=color)
