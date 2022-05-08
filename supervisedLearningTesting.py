import sys
import os
import utils
import glob
import dataReader
from Tasks import QuarterTask, UndersampleFourierTask, VariableNoiseTask
from model_zoo.udvd_model import UDVD
from model_zoo.dncnn_model import DnCNN
from model_zoo.unet_model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)

# argument variables
task_names = utils.get_task_names()
model_names = utils.get_model_names()
if len(sys.argv) != 6 or sys.argv[1] not in task_names or not sys.argv[2].isnumeric or sys.argv[5] not in model_names:
    sys.exit(f"Usage: supervisedLearningTesting.py [task] [gpu #] [checkpoint_name] [epoch] [model_name] task={task_names} model_name={model_names}")
task_name = sys.argv[1]
checkpoint_name = sys.argv[3]
epoch = sys.argv[4]
model_name = sys.argv[5]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]

# get options
opt = utils.get_options(f"./task_configs/{task_name}_options.json")

# Defining the task to solve
task = utils.get_task(task_name, opt, testing=True)

# creating model
net = utils.get_model(model_name)

criterion = nn.MSELoss()

if torch.cuda.is_available():
    net.cuda()
    criterion.cuda()

# load the checkpoint
checkpoint = torch.load(f'./runs/{checkpoint_name}/net{epoch}.pth')
net.load_state_dict(checkpoint['net'])

# Getting data
data_path = os.path.join("data", opt["data_folder_name"])
h5_files = glob.glob(os.path.join(data_path, "*.h5"))
h5_files_test = utils.get_testing_data(h5_files)
#h5_files_train, h5_files_val = utils.train_val_split(h5_files, opt["train_val_split"])
# Getting max pixel value of all data to normalize data
# max_pixel_val = 0
# for j in range(len(h5_files)):
#     data_loader = dataReader.get_dataloader(h5_files[j], 'reconstruction_rss', opt["batch_size"])
#     for i, data in enumerate(data_loader):
#         max_pixel_val = max(max_pixel_val, torch.max(data))

# testing each image from "validation set"
out_folder = checkpoint_name+f"_epoch{epoch}"
psnr = []
ssim = []
step = 0
for k in range(len(h5_files_test)):
        data_loader = dataReader.get_dataloader(h5_files_test[k], 'reconstruction_rss', batch_size=1)

        for l, data in enumerate(data_loader):
            step += 1

            ground_truth = utils.preprocess(data)
            #print(ground_truth)
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth, seed=step)
            #####################################
            net.eval()
            if model_name in ["udvd", "udvd_abl"]:
                y_pred = net(inputs, kernel, noise)
            else:
                y_pred = net(inputs)
            # loss = criterion(y_pred, ground_truth)

            for p in net.parameters():
                print (p.data[0][0][0][0])
                break
            
            with torch.no_grad():
                y_pred = torch.clamp(y_pred, 0., 1.)
                utils.save_image(ground_truth, out_folder, str(step)+"_groundtruth")
                utils.save_image(inputs, out_folder, str(step)+"_noisy")
                utils.save_image(y_pred, out_folder, str(step)+"_reconstructed")
                psnr.append(utils.batch_PSNR(ground_truth, y_pred, 1))
                #ssim.append(1)
                ssim.append(utils.get_ssim(ground_truth, y_pred))
            # batch_psnr = utils.batch_PSNR(y_pred, ground_truth, 1)
            # writer.add_scalar("val_loss", loss.item(), epoch)


utils.write_test_file(psnr, ssim, out_folder)
