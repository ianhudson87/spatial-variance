import sys
import os
import utils
import glob
import dataReader
import Tasks.UndersampleFourierTask as UF
import Tasks.VariableNoiseTask as VN
import model
import torch
import torch.nn as nn
import torch.optim as optim

# argument variables
task_names = ["undersample", "vnoise"]
print(sys.argv)
print(type(sys.argv[1]))
print('vnoise' in task_names)
if len(sys.argv) != 5 or sys.argv[1] not in task_names or not sys.argv[2].isnumeric:
    sys.exit("Usage: testing.py [task] [gpu #] [checkpoint_name] [epoch] task={undersample, vnoise}")
task_name = sys.argv[1]
checkpoint_name = sys.argv[3]
epoch = sys.argv[4]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]

# get options
opt = utils.get_options(f"./task_configs/{task_name}_options.json")
opt["out_folder"] = "./runs/" + opt["model_name"] + utils.get_date_time()

# Defining the task to solve
task_index = task_names.index(task_name)
if task_index==0:
    task = UF.UndersampleFourierTask(opt["sample_percent"])
elif task_index==1:
    task = VN.VariableNoiseTask(20, 50, 20)

# creating model
net = model.UDVD(k=5, in_channels=1, depth=5)
criterion = nn.MSELoss()

# load the checkpoint
checkpoint = torch.load(f'./logs/{checkpoint_name}/net{epoch}.pth')
net.load_state_dict(checkpoint['net'])

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

# testing each image from "validation set"
step = 0
for k in range(len(h5_files_val)):
        data_loader = dataReader.get_dataloader(h5_files_val[k], 'reconstruction_rss', batch_size=1)

        for l, data in enumerate(data_loader):
            step += 1
            ground_truth = torch.unsqueeze(data, 1)/max_pixel_val # add channel dimension to data
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth)
            #####################################

            net.eval()
            y_pred = net(inputs, kernel, noise)
            # loss = criterion(y_pred, ground_truth)

            y_pred = torch.clamp(y_pred, 0., 1.)
            utils.save_image(ground_truth, checkpoint_name, str(step)+"_groundtruth")
            utils.save_image(inputs, checkpoint_name, str(step)+"_noisy")
            utils.save_image(y_pred, checkpoint_name, str(step)+"_reconstructed")
            # batch_psnr = utils.batch_PSNR(y_pred, ground_truth, 1)
            # writer.add_scalar("val_loss", loss.item(), epoch)