import numpy as np
import torch

from utils import fft2, ifft2, fft2_c, ifft2_c


class NMR_two_channel:
    def __init__(self, mask):
        self.mask = mask

    def A(self, x):
        return self.mask * ifft2(x)

    def AT(self, x):
        return fft2(x)

    def A_dagger(self, x):
        return self.AT(x)


class NMR_complex:
    def __init__(self, mask):
        self.mask = mask

    def A(self, x):
        return ifft2_c(x) * self.mask

    def AT(self, x):
        return fft2_c(x)

    def A_dagger(self, x):
        return self.AT(x)

