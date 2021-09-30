import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio

def imshow(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def get_options(options_file = "options.json"):
    with open(options_file) as json_file:
        data = json.load(json_file)
        return data

def train_val_split(files, train_frac):
    split_point = int(len(files)*train_frac)
    print("SPLIT POINT", split_point)
    return files[:split_point], files[split_point:]

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        # PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])