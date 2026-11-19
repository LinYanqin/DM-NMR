import numpy as np
import torch

from utils import fft2, ifft2


class NMR_two_channel:
    def __init__(self, mask):
        self.mask = mask

    def A(self, x):
        return self.mask * ifft2(x)

    def AT(self, x):
        return fft2(x)

    def A_dagger(self, x):
        return self.AT(x)



