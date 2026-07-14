import glob

from torch.utils.data import Dataset, DataLoader
import numpy as np

y_axis = 64
x_aixs = 64


def load_data(
    *,
    data_dir,
    batch_size,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    train_path = data_dir + '/Label'
    train_dataset = abs_2D_train(train_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    while True:
        yield from train_loader


class abs_2D_train(Dataset):
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
    data_temp = np.reshape(data_temp, (y_axis, x_aixs, 4), 'F')
    data_temp = np.transpose(data_temp, [2, 0, 1])
    # data_temp = ifft2(data_temp)
    # data_temp = np.asarray(data_temp, dtype=np.float32)
    # mask = np.zeros((4, y_axis, x_aixs), dtype=np.float32)
    # mask[:, :20, :20] = 1
    # data = data_temp * mask
    data = data_temp
    out_dict = {}
    return data, out_dict



def real_to_nchw_comp(x):
    """
    [2, 320, 320] real --> [1, 320, 320] comp
    """
    x = x[0:1, :, :] + x[1:2, :, :] * 1j
    return x

def nchw_comp_to_real(x):
    """
    [1, 320, 320] comp --> [2, 320, 320] real
    """
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real, x_imag], axis=0)

def ifftshift(x):
    s0 = (x.shape[-2] // 2)  # 32
    s1 = (x.shape[-1] // 2)  # 32
    x = np.concatenate([x[:, s0:, :], x[:, :s0, :]], axis=-2)  # exchange down and up
    x = np.concatenate([x[:, :, s1:], x[:, :, :s1]], axis=-1)  # exchange right and left
    return x

def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  x_1 = x[0:2, :, :]
  x_2 = x[2:4, :, :]
  complex_tensor_1 = real_to_nchw_comp(x_1)
  ifft_result_1 = np.fft.ifft2(ifftshift(complex_tensor_1), axes=(-2, -1))
  real_tensor_1 = nchw_comp_to_real(ifft_result_1)
  complex_tensor_2 = real_to_nchw_comp(x_2)
  ifft_result_2 = np.fft.ifft2(ifftshift(complex_tensor_2), axes=(-2, -1))
  real_tensor_2 = nchw_comp_to_real(ifft_result_2)
  return np.concatenate([real_tensor_1, real_tensor_2], axis=0)
