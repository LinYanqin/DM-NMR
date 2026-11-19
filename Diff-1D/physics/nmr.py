from utils import fft, ifft

class NMR_two_channel_freq:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def A(self, x):
        return ifft(x) * self.mask

    def AT(self, x):
        return fft(x * self.mask)

    def A_dagger(self, x):
        return self.AT(x)
