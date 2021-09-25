import os
import glob
import dataReader
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import sampleMask
import utils

data_folder_name = "singlecoil_test_v2"
data_path = os.path.join("data", data_folder_name)

h5_files = glob.glob(os.path.join(data_path, "*.h5"))
# print(h5_files[0])

data_loader = dataReader.get_dataloader(h5_files[0])

ifftshift = torch.fft.ifftshift
ifft2 = torch.fft.ifft2
ifftn = torch.fft.ifftn
ifft = torch.fft.ifft
fftshift = torch.fft.fftshift

for i, data in enumerate(data_loader):
    # data is batch
    for j in range(data.shape[0]):
        kspace = data[0]
        sample_mask = sampleMask.get_sample_mask(kspace.shape[0], kspace.shape[1], 0.5)
        # print(kspace.shape)
        # print(sample_mask.shape)
        undersampled_kspace = kspace * sample_mask
        print(undersampled_kspace.dtype)

        image = torch.absolute(ifftshift(ifft2(fftshift(kspace))))
        undersampled_image = torch.absolute(ifftshift(ifft2(fftshift(undersampled_kspace))))

        # test = torch.absolute(ifft2(kspace))
        # print(sample_mask)
        utils.imshow(sample_mask)

        utils.imshow(image)

        utils.imshow(undersampled_image)

    # print(data.shape)
    # print(data[0].shape)
    # np_img = np.array(data[0])
    # # print(np_img)
    # plt.figure()
    # plt.imshow(np.array(data[0]))
    if i>20:
        break

# f = h5py.File(h5_files[0], 'r')
# cv2.imshow(data)