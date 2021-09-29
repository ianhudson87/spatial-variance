import os
import h5py
import torch
from torchvision import transforms
import random
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_file, img_type):
        # img_type = 'kspace', 'reconstruction_esc', 'reconstruction_rss'
        super(Dataset, self).__init__()
        # save train file name
        self.train_file = train_file
        self.img_type = img_type

        # get number of images
        h5f = h5py.File(self.train_file, 'r')
        print("here", h5f.keys())
        h5d = h5f[self.img_type] # h5 dataset of the images
        self.num_images = h5d.len()
        h5f.close()

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        h5f = h5py.File(self.train_file, 'r')
        h5d = h5f[self.img_type]
        # print(type(h5d[0]))
        data_np = h5d[index] # np array (dtype=complex64)
        # print(data_np.dtype)
        data_torch = torch.from_numpy(data_np)
        h5f.close()
        return data_torch

def get_dataloader(h5_path, img_type, batch_size):
    # create a torch dataloader from a h5 file
    dataset = Dataset(train_file=h5_path, img_type=img_type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader