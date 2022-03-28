from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2

def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c

M = 32
J = 2
L = 4
filters_set = filter_bank(M, M, J, L=L)

#high pass
fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
fig.set_figheight(6)
fig.set_figwidth(6)
i = 0
for filter in filters_set['psi']:
    f = filter[0]
    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    axs[i // L, i % L].imshow(colorize(filter_c))
    axs[i // L, i % L].axis('off')
    i = i+1

fig.show()

'''
# low pass
plt.figure()
plt.axis('off')
plt.set_cmap('gray_r')

f = filters_set['phi'][0]

filter_c = fft2(f)
filter_c = np.fft.fftshift(filter_c)
filter_c = np.abs(filter_c)
plt.imshow(filter_c)
plt.show()
'''