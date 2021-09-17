import os
import h5py
import torch
from torchvision import transforms
import random
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_file):
        super(Dataset, self).__init__()
        # save train file name
        self.train_file = train_file

        # get number of images
        h5f = h5py.File(self.train_file, 'r')
        h5d = h5f['kspace'] # h5 dataset of the images
        self.num_images = h5d.len()
        h5f.close()

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        h5f = h5py.File(self.train_file, 'r')
        h5d = h5f['kspace']
        # print(type(h5d[0]))
        data_np = h5d[index] # np array (dtype=complex64)
        # print(data_np.dtype)
        data_torch = torch.from_numpy(data_np)
        h5f.close()
        return data_torch

def get_dataloader(h5_path):
    # create a torch dataloader from a h5 file
    dataset = Dataset(train_file=h5_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    return loader