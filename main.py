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
from tensorboardX import SummaryWriter
import utils
from model_zoo.udvd_model import UDVD
from model_zoo.dncnn_model import DnCNN
from model_zoo.unet_model import UNet
import sys
import Tasks.UndersampleFourierTask as UF
import Tasks.VariableNoiseTask as VN
torch.set_num_threads(1)
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())

# argument variables
task_names = ["undersample", "vnoise"]
model_names = ["udvd", "dncnn", "unet"]
if len(sys.argv) != 4 or sys.argv[1] not in task_names or not sys.argv[2].isnumeric or sys.argv[3] not in model_names:
    sys.exit("Usage: main.py [task] [gpu #] [model] task={undersample, vnoise} model={udvd, dncnn, unet}")
task_name = sys.argv[1]
model_name = sys.argv[3]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]

# get options
opt = utils.get_options(f"./task_configs/{task_name}_options.json")
opt["out_folder"] = os.path.join("runs", f"{model_name}_{opt['task_name']}_{utils.get_date_time()}")

# Defining the task to solve
task_index = task_names.index(task_name)
if task_index==0:
    task = UF.UndersampleFourierTask(opt["sample_percent"])
elif task_index==1:
    task = VN.VariableNoiseTask(20, 50, 20)

# creating model
if model_name == "udvd":
    print("Using UDVD model")
    net = UDVD(k=5, in_channels=1, depth=5)
elif model_name == "dncnn":
    print("Using DNCNN model")
    net = DnCNN(channels=1)
elif model_name == "unet":
    print("Using UNet model")
    net = UNet(in_channels=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=opt["lr"])

if torch.cuda.is_available():
    net.cuda()
    criterion.cuda()

# Tensorboard
writer = SummaryWriter(log_dir=opt["out_folder"])

# Getting data
data_path = os.path.join("data", opt["data_folder_name"])
h5_files = glob.glob(os.path.join(data_path, "*.h5"))
h5_files_train, h5_files_val = utils.train_val_split(h5_files, opt["train_val_split"])
# Getting max pixel value of all data to normalize data
max_pixel_val = 0
for j in range(len(h5_files)):
    data_loader = dataReader.get_dataloader(h5_files[j], 'reconstruction_rss', opt["batch_size"])
    for i, data in enumerate(data_loader):
        max_pixel_val = max(max_pixel_val, torch.max(data))
    

step=0

# Training
for epoch in range(opt["epochs"]):
    for j in range(len(h5_files_train)):
        # getting data loader
        data_loader = dataReader.get_dataloader(h5_files_train[j], 'reconstruction_rss', opt["batch_size"])
        
        # training
        for i, data in enumerate(data_loader):
            step += 1
            
            ground_truth = utils.preprocess(data)
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            # data is batch of ground truth images

            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth)
            inputs = torch.clamp(inputs, 0., 1.)
            #####################################

            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            if model_name == "udvd":
                y_pred = net(inputs, kernel, noise)
            elif model_name in ["dncnn", "unet"]:
                y_pred = net(inputs)
            loss = criterion(y_pred, ground_truth)
            loss.backward()
            optimizer.step()

            y_pred = torch.clamp(y_pred, 0., 1.)
            batch_psnr = utils.batch_PSNR(y_pred, ground_truth, 1)
            writer.add_scalar("train_loss", loss.item(), step)
            writer.add_scalar("batch_psnr", batch_psnr, step)
            print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}, psnr: {batch_psnr}")

    # Validation 
    total_psnr = 0
    batches = 1
    for k in range(len(h5_files_val)):
        data_loader = dataReader.get_dataloader(h5_files_val[k], 'reconstruction_rss', opt["batch_size"])
        
        for l, data in enumerate(data_loader):
            ground_truth = torch.unsqueeze(data, 1)/max_pixel_val # add channel dimension to data
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth)
            #####################################

            net.eval()
            if model_name == "udvd":
                y_pred = net(inputs, kernel, noise)
            elif model_name == "dncnn":
                y_pred = net(inputs)
            # loss = criterion(y_pred, ground_truth)

            y_pred = torch.clamp(y_pred, 0., 1.)
            batch_psnr = utils.batch_PSNR(y_pred, ground_truth, 1)
            total_psnr += batch_psnr
            batches += 1
            # writer.add_scalar("val_loss", loss.item(), epoch)
    writer.add_scalar("val_psnr", total_psnr / batches, epoch)
    print(f"Validation: epoch: {epoch}, psnr: {total_psnr / batches}")

    
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(opt["out_folder"], 'net%d.pth' % (epoch+1)))
