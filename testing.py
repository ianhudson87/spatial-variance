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
torch.set_num_threads(1)

# argument variables
task_names = ["undersample", "vnoise"]
model_names = ["udvd", "dncnn"]
if len(sys.argv) != 6 or sys.argv[1] not in task_names or not sys.argv[2].isnumeric or sys.argv[5] not in model_names:
    sys.exit("Usage: testing.py [task] [gpu #] [checkpoint_name] [epoch] [model_name] task={undersample, vnoise} model_name={udvd, dncnn}")
task_name = sys.argv[1]
checkpoint_name = sys.argv[3]
epoch = sys.argv[4]
model_name = sys.argv[5]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]

# get options
opt = utils.get_options(f"./task_configs/{task_name}_options.json")
opt["out_folder"] = "./runs/" + model_name + opt["task_name"] + utils.get_date_time()

# Defining the task to solve
task_index = task_names.index(task_name)
if task_index==0:
    task = UF.UndersampleFourierTask(opt["sample_percent"])
elif task_index==1:
    task = VN.VariableNoiseTask(20, 50, 20)

# creating model
if model_name == "udvd":
    print("Using UDVD model")
    net = model.UDVD(k=5, in_channels=1, depth=5)
elif model_name == "dncnn":
    print("Using DNCNN model")
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
# Getting max pixel value of all data to normalize data
# max_pixel_val = 0
# for j in range(len(h5_files)):
#     data_loader = dataReader.get_dataloader(h5_files[j], 'reconstruction_rss', opt["batch_size"])
#     for i, data in enumerate(data_loader):
#         max_pixel_val = max(max_pixel_val, torch.max(data))

# testing each image from "validation set"
step = 0
for k in range(len(h5_files_test)):
        data_loader = dataReader.get_dataloader(h5_files_test[k], 'reconstruction_rss', batch_size=1)

        for l, data in enumerate(data_loader):
            step += 1
            #print(data.shape)
            #print(data)
            #max_vals = torch.max(data) # max vals for each image
            #print(max_vals)
            #data = data / max_vals
            ground_truth = torch.unsqueeze(data, 1) # add channel dimension to data
            max_vals = torch.amax(ground_truth, dim=(2, 3))
            ground_truth = ground_truth / max_vals
            #print(ground_truth)
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            #####################################
            # Applying deconstruction to image
            inputs, kernel, noise = task.get_deconstructed(ground_truth)
            inputs = torch.clamp(inputs, 0., 1.)
            #####################################
            net.eval()
            if model_name == "udvd":
                y_pred = net(inputs, kernel, noise)
            elif model_name == "dncnn":
                y_pred = net(inputs)
            # loss = criterion(y_pred, ground_truth)
            
            with torch.no_grad():
                y_pred = torch.clamp(y_pred, 0., 1.)
                utils.save_image(ground_truth, checkpoint_name, str(step)+"_groundtruth")
                utils.save_image(inputs, checkpoint_name, str(step)+"_noisy")
                utils.save_image(y_pred, checkpoint_name, str(step)+"_reconstructed")
            # batch_psnr = utils.batch_PSNR(y_pred, ground_truth, 1)
            # writer.add_scalar("val_loss", loss.item(), epoch)
