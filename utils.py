import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datetime import datetime
import cv2
import os
from pathlib import Path

def imshow(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def get_options(options_file = "options.json"):
    with open(options_file) as json_file:
        data = json.load(json_file)
        return data

def train_val_split(files, train_frac):
    split_point = int((len(files)-1)*train_frac) # save one file for testing
    print("SPLIT POINT", split_point)
    return files[:split_point], files[split_point:-1]

def get_testing_data(files):
    return [files[-1]]

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
    return str(dateTimeObj.month) + "-" + str(dateTimeObj.day) + "-" + str((dateTimeObj.hour-5)%24) + "-" + str(dateTimeObj.minute)

def save_image(img, folder, name):
    # for greyscale image only, batch_size=1
    path = os.path.join("./test_logs", folder)
    Path(path).mkdir(parents=True, exist_ok=True) # create directory if doesn't exist
    np_image = np.array(img.cpu())[0]
    np_image = np.swapaxes(np.swapaxes(np_image, 0, 2), 0, 1)*256
    cv2.imwrite(os.path.join(path, name+".tif"), np_image)

def get_psnr(img_true, img_test):
    img_true = img_true.cpu().numpy().astype(np.float32)[0]
    img_test = img_test.cpu().numpy().astype(np.float32)[0]
    return peak_signal_noise_ratio(img_true, img_test, data_range=1.)

def get_ssim(img_true, img_test):
    img_true = np.swapaxes(img_true.cpu().numpy().astype(np.float32)[0], 0, 2)
    img_test = np.swapaxes(img_test.cpu().numpy().astype(np.float32)[0], 0, 2)
    return structural_similarity(img_true, img_test, multichannel=True)

def float_str(x, places):
    return ("{:." + str(places) + "f}").format(x)

def write_test_file(psnr_vals, ssim_vals, folder):
    path = os.path.join("test_logs", folder, "_stats.txt")
    f = open(path, "w")
    f.write("psnr, ssim" + "\n")
    for k in range(len(psnr_vals)):
        f.write(float_str(psnr_vals[k], 2) + "," + float_str(ssim_vals[k], 4))
        f.write("\n")
    f.close()