import os, glob
import dataReader, utils
import copy
import torch
import torch.linalg
from torch.autograd import Variable

data_path = os.path.join("data", "singlecoil_val")
iterations = 100
batch_size = 1
mri_image_type = 'reconstruction_rss'
task_name = "undersample"
prior_weight = 1
learning_rate = 0.0001
os.environ["CUDA_VISIBLE_DEVICES"]=""


opt = utils.get_options(f"./task_configs/{task_name}_options.json")

task = utils.get_task(task_name, opt)

h5_files = glob.glob(os.path.join(data_path, "*.h5"))
print(h5_files[1])

for h5_file in h5_files:
    data_loader = dataReader.get_dataloader(h5_file, mri_image_type, batch_size)

    for i, data in enumerate(data_loader):
        ground_truth = utils.preprocess(data)

        deconstructed, kernel, noise = task.get_deconstructed(ground_truth)
        # deconstructed = y = given measurement

        prediction = copy.copy(deconstructed)
        var_prediction = Variable(prediction, requires_grad=True)
        # prediction = x = current guess

        for iteration in range(iterations):
            difference = var_prediction - deconstructed
            squared_norm_difference = torch.sum(difference * difference)
            # print("snd", squared_norm_difference)
            #TODO: check if difference is supposed to be in k-space

            prior = torch.sum(var_prediction * var_prediction)
            # TODO: change this later

            error = squared_norm_difference + (prior_weight * prior)
            error.backward() # calculate gradients


            # subtract gradient of error wrt prediction from the prediction
            var_prediction = Variable(var_prediction - (learning_rate * var_prediction.grad), requires_grad=True)
            # This also clears the gradients from var_prediction
        break
    break

print("deconstructed", deconstructed)
print("prediction", var_prediction)

    