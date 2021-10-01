import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from datetime import datetime

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

def get_date_time():
    dateTimeObj = datetime.now()
<<<<<<< HEAD
    str(dateTimeObj.month) + "-" + str(dateTimeObj.day) + "--" + str((dateTimeObj.hour-5)%24) + "-" + str(dateTimeObj.minute)
=======
    return str(dateTimeObj.month) + "-" + str(dateTimeObj.day) + "-" + str((dateTimeObj.hour-5)%24) + "-" + str(dateTimeObj.minute)
>>>>>>> 427e5fd3f09a8975f0fd2a4bce70011c92e4e8a0
