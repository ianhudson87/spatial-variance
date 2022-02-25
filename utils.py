import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datetime import datetime
import cv2
import os
from pathlib import Path
import torch
from Tasks import QuarterTask, UndersampleFourierTask, VariableNoiseTask
from model_zoo.udvd_model import UDVD
from model_zoo.dncnn_model import DnCNN
from model_zoo.unet_model import UNet
from model_zoo.udvd_ablation_model import UDVDablation_nodynamic
from model_zoo.dncnn_ablation_model import DnCNNablationTail, DnCNNablationFull, DnCNNablationHead, DnCNNablationMiddle, DnCNNablation_more_dyn
from model_zoo.dncnn_specnorm_model import DnCNNSpecNorm

def imshow(img, swap_axes=False):
    if swap_axes:
        img = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
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
    print(f"USING TESTING FILE: {files[-1]}")
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

def write_test_file(psnr_vals, ssim_vals, folder, file_name=None):
    if file_name:
        path = os.path.join("test_logs", folder, f"_{file_name}.txt")
    else:
        path = os.path.join("test_logs", folder, "_stats.txt")
    f = open(path, "w")
    f.write("psnr, ssim" + "\n")
    for k in range(len(psnr_vals)):
        f.write(float_str(psnr_vals[k], 2) + "," + float_str(ssim_vals[k], 4))
        f.write("\n")
    f.close()

def preprocess(data):
    # print("new")
    min_vals = torch.amin(data, dim=(1,2))
    min_vals = torch.reshape(min_vals, (data.shape[0], 1, 1))
    data = torch.sub(data, min_vals)

    # print("data", data)
    # print("mins", torch.amin(data, dim=(1,2)))

    max_vals = torch.amax(data, dim=(1,2))
    max_vals = torch.reshape(max_vals, (data.shape[0], 1, 1))
    data = torch.div(data, max_vals)

    # print("data", data)
    # print("mins", torch.amin(data, dim=(1,2)))
    # print("maxs", torch.amax(data, dim=(1,2)))
    
    ground_truth = torch.unsqueeze(data, 1) # add channel dimension to data
    return ground_truth

def get_task(task_name, opt, testing=False):
    task_names = get_task_names()
    task_index = task_names.index(task_name)
    if task_index==0:
        return UndersampleFourierTask.Task(opt["sample_percent"], testing)
    elif task_index==1:
        return VariableNoiseTask.Task(opt["min_stdev"], opt["max_stdev"], opt["patch_size"], testing)
    elif task_index==2:
        # print(opt)
        return QuarterTask.Task((opt["quadrant1_stdev"], opt["quadrant2_stdev"], opt["quadrant3_stdev"], opt["quadrant4_stdev"]), testing)
    else:
        raise ValueError("couldn't find task:", task_name)

def get_model(model_name):
    model_names = get_model_names()
    model_index = model_names.index(model_name)
    if model_index == 0:
        return UDVD(k=5, in_channels=1, depth=5)
    elif model_index == 1:
        return DnCNN(channels=1)
    elif model_index == 2:
        return UNet(in_channels=1)
    elif model_index == 3:
        return UDVDablation_nodynamic(k=5, in_channels=1, depth=5)
    elif model_index == 4:
        return DnCNNablationHead(channels=1)
    elif model_index == 5:
        return DnCNNablationMiddle(channels=1)
    elif model_index == 6:
        return DnCNNablationTail(channels=1)
    elif model_index == 7:
        return DnCNNablationFull(channels=1)
    elif model_index == 8:
        return DnCNNablation_more_dyn(channels=1)
    elif model_index == 9:
        return DnCNNSpecNorm(1)
    else:
        raise ValueError("couldn't find model:", model_name)

def get_model_names():
    return ["udvd", "dncnn", "unet", "udvd_abl_nodyn", "dncnn_abl_head", "dncnn_abl_mid", "dncnn_abl_tail", "dncnn_abl_full", "dncnn_abl_moredyn", "dncnn_spec"]

def get_task_names():
    return ["undersample", "vnoise", "quarter"]
