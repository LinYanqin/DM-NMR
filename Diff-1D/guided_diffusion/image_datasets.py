import glob

from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio

y_axis = 256


def load_1DtonD_data(
        *,
        data_dir,
        batch_size,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    train_path = data_dir
    train_dataset = train_1DtonD(train_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    while True:
        yield from train_loader


class train_1DtonD(Dataset):
    def __init__(self, train_path):
        self.label = sio.loadmat(train_path)
        self.label = self.label['label_data']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data_temp = self.label[idx]
        data_temp = np.squeeze(data_temp)
        data_temp = np.asarray(data_temp, dtype=np.float32)
        data_temp = np.transpose(data_temp, [1, 0])
        data = data_temp
        out_dict = {}
        return data, out_dict


def load_data(
    *,
    data_dir,
    batch_size,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    train_path = data_dir + '/Label'
    train_dataset = abs_1D_train(train_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    while True:
        yield from train_loader


class abs_1D_train(Dataset):
  def __init__(self, root, sort=True):
    self.root = root
    self.data_list = glob.glob(f'{root}/*.txt')
    if sort:
      self.data_list = sorted(self.data_list)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    fr = open(fname)
    data_temp = []
    for line in fr.readlines():
      lineArr = line.strip().split()
      data_temp.append(lineArr[:])
    data_temp = np.asarray(data_temp, dtype=np.float32)
    data_temp = np.reshape(data_temp, (y_axis, 2), 'F')
    data_temp = np.transpose(data_temp, [1, 0])
    data_temp = ifft(data_temp)
    data_temp = np.asarray(data_temp, dtype=np.float32)
    mask = np.zeros((2,y_axis),dtype=np.float32)
    mask[:,:64]=1
    data = data_temp *mask
    out_dict = {}
    return data, out_dict


def nchw_comp_to_real(x):
    """
    [1, 320] comp --> [2, 320] real
    """
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real, x_imag], axis=0)

def real_to_nchw_comp(x):
    """
    [2, 320] real --> [1, 320] comp
    """
    x = x[0:1, :] + x[1:2, :] * 1j
    return x


def ifft(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  complex_tensor = real_to_nchw_comp(x)
  ifft_result = np.fft.ifft(complex_tensor, axis=-1)
  return nchw_comp_to_real(ifft_result)
