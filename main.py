import os
import glob
import dataReader
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sampleMask
import utils
import model


opt = utils.get_options()

# defining fft
ifftshift = torch.fft.ifftshift
ifft2 = torch.fft.ifft2
ifftn = torch.fft.ifftn
ifft = torch.fft.ifft
fft = torch.fft.fft
fftshift = torch.fft.fftshift

# creating model
net = model.UDVD(k=5, r=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=opt["lr"], amsgrad=True)

data_path = os.path.join("data", opt["data_folder_name"])
h5_files = glob.glob(os.path.join(data_path, "*.h5"))
for j in range(len(h5_files)):
    # getting data loader
    data_loader = dataReader.get_dataloader(h5_files[j], 'reconstruction_rss', opt["batch_size"])

    # training
    for i, data in enumerate(data_loader):
        # data is batch
        batch_kspace = fft(data)
        h = batch_kspace.shape[1]
        w = batch_kspace.shape[2]
        batch_sample_mask = sampleMask.get_batch_sample_mask(h, w, opt["sample_percent"], opt["batch_size"])
        
        undersampled_batch_kspace = batch_kspace * batch_sample_mask

        undersampled_batch_image = torch.absolute(ifft(undersampled_batch_kspace))

        inputs = undersampled_batch_image
        kernel = torch.zeros(opt["batch_size"], 15, 64, 64)
        noise = torch.zeros(opt["batch_size"], 1, 64, 64)
        print(inputs.shape)


        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        y_pred = net(inputs, kernel, noise)
        loss = criterion(y_pred, data)
        loss.backward()
        optimizer.step()
        # for j in range(data.shape[0]):
        #     ground_truth = data[j]
        #     kspace = fft(ground_truth)
        #     h = kspace.shape[0]
        #     w = kspace.shape[1]
        #     sample_mask = sampleMask.get_sample_mask(h, w, opt["sample_percent"])

        #     undersampled_kspace = kspace * sample_mask
        #     undersampled_ground_truth = torch.absolute(ifft(undersampled_kspace))

        #     print("new image")
        #     utils.imshow(sample_mask)
        #     utils.imshow(batch_sample_mask[j])
        #     utils.imshow(ground_truth)
        #     utils.imshow(data[j])
        #     utils.imshow(undersampled_ground_truth)
        #     utils.imshow(undersampled_batch_image[j])

# f = h5py.File(h5_files[0], 'r')
# cv2.imshow(data)