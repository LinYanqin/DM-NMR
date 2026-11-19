import glob

from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio

y_axis = 256

def load_data(
    *,
    data_dir,
    batch_size,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    train_path = data_dir + '/Label'
    train_dataset = ri_1D_train(train_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    while True:
        yield from train_loader


class ri_1D_train(Dataset):
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
    data = data_temp
    out_dict = {}
    return data, out_dict

