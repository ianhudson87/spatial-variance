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
import model
import Tasks.UndersampleFourierTask as UF
import Tasks.VariableNoiseTask as VN
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())


opt = utils.get_options()
opt["out_folder"] = "./runs/" + utils.get_date_time()


# creating model
net = model.UDVD(k=5, in_channels=1, depth=5)
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

# Defining the task to solve
task = UF.UndersampleFourierTask(opt["sample_percent"])
# task = VN.VariableNoiseTask(20, 50, 20)

step=0

# Training
for epoch in range(opt["epochs"]):
    for j in range(len(h5_files_train)):
        # getting data loader
        data_loader = dataReader.get_dataloader(h5_files_train[j], 'reconstruction_rss', opt["batch_size"])

        # training
        for i, data in enumerate(data_loader):
            step += 1
            ground_truth = torch.unsqueeze(data, 1)/max_pixel_val # add channel dimension to data, apply normalization across all data
            # data is batch of ground truth images

            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth)
            #####################################

            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            y_pred = net(inputs, kernel, noise)
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
    batches = 0
    for k in range(len(h5_files_val)):
        data_loader = dataReader.get_dataloader(h5_files_val[k], 'reconstruction_rss', opt["batch_size"])

        for l, data in enumerate(data_loader):
            ground_truth = torch.unsqueeze(data, 1)/max_pixel_val # add channel dimension to data

            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth)
            #####################################

            net.eval()
            y_pred = net(inputs, kernel, noise)
            # loss = criterion(y_pred, ground_truth)

            y_pred = torch.clamp(y_pred, 0., 1.)
            batch_psnr = utils.batch_PSNR(y_pred, ground_truth, 1)
            batches += 1
            # writer.add_scalar("val_loss", loss.item(), epoch)
    writer.add_scalar("val_psnr", total_psnr / batches, epoch)
    print(f"Validation: epoch: {epoch}, psnr: {total_psnr / batches}")

    
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(opt["out_folder"], 'net%d.pth' % (epoch+1)))
