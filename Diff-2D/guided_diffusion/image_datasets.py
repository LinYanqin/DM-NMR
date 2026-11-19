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
    train_dataset = ri_2D_train(train_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    while True:
        yield from train_loader


class ri_2D_train(Dataset):
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
    data = data_temp
    out_dict = {}
    return data, out_dict

