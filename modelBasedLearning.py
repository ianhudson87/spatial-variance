import os, glob
import dataReader, utils
import copy
import torch
import torch.linalg
from torch.autograd import Variable
import sampleMask

data_path = os.path.join("data", "singlecoil_val")
iterations = 1
batch_size = 1
mri_image_type = 'reconstruction_rss'
task_name = "undersample"
prior_weight = 0.001
learning_rate = 0.001
os.environ["CUDA_VISIBLE_DEVICES"]="4"
model_name = "dncnn"
checkpoint_name = "dncnn_Quarter_11-23-7-45"
epoch = 50
out_folder = "modelbased_learning" + str(1)

# initialize network
net = utils.get_model(model_name)
if torch.cuda.is_available():
    net.cuda()

# load the checkpoint
checkpoint = torch.load(f'./runs/{checkpoint_name}/net{epoch}.pth')
net.load_state_dict(checkpoint['net'])

opt = utils.get_options(f"./task_configs/{task_name}_options.json")

task = utils.get_task(task_name, opt)

# Getting data
data_path = os.path.join("data", opt["data_folder_name"])
h5_files = glob.glob(os.path.join(data_path, "*.h5"))
h5_files_test = utils.get_testing_data(h5_files)

def methodA(var_prediction, given_measurement, iterations):
    # var_prediction = x^t
    # given_measurement = y
    # returns: final prediction (x^(t_final))
    
    for iteration in range(iterations):
        var_prediction_kspace = torch.fft.fft(var_prediction)
        # prediction_kspace = F(x^t)

        var_prediction_kspace_undersampled = sampleMask.sample_image(var_prediction_kspace, opt["sample_percent"])
        # TODO: this should be random mask using same percentage?
        # prediction_kspace_undersampled = MF(x^t) = H(x^t)

        difference = var_prediction_kspace_undersampled - given_measurement
        # difference = H(x^t) - y

        absolute_difference = torch.abs(difference)
        # take absolute value of each element so that complex values become real values (so that we can take the norm)

        # squared_norm_difference = torch.sum(difference * difference)
        squared_norm_difference = torch.square(torch.norm(absolute_difference, p=2))
        # squared_norm_difference = g(x^t)

        prior = torch.square(torch.norm(var_prediction))
        # prior = R(x^t)
        # TODO: change this later

        error = squared_norm_difference + (prior_weight * prior)
        # error = (prior_weight * prior)
        # error = g(x^t) + R(x^t) = f(x^t)

        error.backward() # calculate gradients
        # var_prediction.grad now contains gradient of f(x^t) (wrt x^t)

        # subtract gradient of error wrt prediction from the prediction
        var_prediction = Variable(var_prediction - (learning_rate * var_prediction.grad), requires_grad=True)
        # This also clears the gradients from var_prediction
    return var_prediction

def methodB(var_prediction, given_measurement, iterations):
    # assumes prior is L-2 norm squared
    # var_prediction = x^t
    # given_measurement = y
    # returns: final prediction (x^(t_final))
    
    for iteration in range(iterations):
        ######################### STEP 1 #########################

        var_prediction_kspace = torch.fft.fft(var_prediction)
        # prediction_kspace = F(x^t)

        var_prediction_kspace_undersampled = sampleMask.sample_image(var_prediction_kspace, opt["sample_percent"])
        # TODO: this should be random mask using same percentage?
        # prediction_kspace_undersampled = MF(x^t) = H(x^t)

        difference = var_prediction_kspace_undersampled - given_measurement
        # difference = H(x^t) - y

        absolute_difference = torch.abs(difference)
        # take absolute value of each element so that complex values become real values (so that we can take the norm)

        # squared_norm_difference = torch.sum(difference * difference)
        squared_norm_difference = torch.square(torch.norm(absolute_difference, p=2))
        # squared_norm_difference = g(x^t)

        squared_norm_difference.backward() # calculate gradients
        # var_prediction.grad now contains gradient of g(x^t) (wrt x^t)

        # subtract gradient of error wrt prediction from the prediction
        intermediate_prediction = var_prediction - (learning_rate * var_prediction.grad)
        # intermediate_prediction = z^(t+1)

        ######################### STEP 2 #########################

        # take the gradient and set to 0. This is our new prediction
        # new prediction = Q = \frac{2}{2 + 2\gamma} z^{t+1}, gamma = prior weight
        var_prediction = Variable( 2/(2 + 2*prior_weight) * intermediate_prediction, requires_grad=True)
        # This also clears the gradients from var_prediction

    return var_prediction

def methodC(var_prediction, given_measurement, iterations, denoising_net):
    # assumes prior is L-2 norm squared
    # var_prediction = x^t
    # given_measurement = y
    # denoising_net = denoiser with pretrained weights
    # returns: final prediction (x^(t_final))
    
    for iteration in range(iterations):
        ######################### STEP 1 #########################

        var_prediction_kspace = torch.fft.fft(var_prediction)
        # prediction_kspace = F(x^t)

        var_prediction_kspace_undersampled = sampleMask.sample_image(var_prediction_kspace, opt["sample_percent"])
        # TODO: this should be random mask using same percentage?
        # prediction_kspace_undersampled = MF(x^t) = H(x^t)

        difference = var_prediction_kspace_undersampled - given_measurement
        # difference = H(x^t) - y

        absolute_difference = torch.abs(difference)
        # take absolute value of each element so that complex values become real values (so that we can take the norm)

        # squared_norm_difference = torch.sum(difference * difference)
        squared_norm_difference = torch.norm(absolute_difference, p=2)
        # squared_norm_difference = g(x^t)

        squared_norm_difference.backward() # calculate gradients
        # var_prediction.grad now contains gradient of g(x^t) (wrt x^t)

        # subtract gradient of error wrt prediction from the prediction
        intermediate_prediction = torch.clamp(var_prediction - (learning_rate * var_prediction.grad), 0., 1.)
        # intermediate_prediction = z^(t+1)

        ######################### STEP 2 #########################

        # Apply denoising to the intermediate prediction
        denoised = torch.clamp(denoising_net(intermediate_prediction), 0., 1.)
        var_prediction = Variable(1 * denoised + 0 * intermediate_prediction, requires_grad=True)
        # This also clears the gradients from var_prediction

    return var_prediction

for k in range(len(h5_files_test)):
    data_loader = dataReader.get_dataloader(h5_files_test[k], 'reconstruction_rss', batch_size=batch_size)

    for i, data in enumerate(data_loader):
        if i != 20: continue

        ground_truth = utils.preprocess(data) # ground truth is pixel space
        utils.save_image(ground_truth, out_folder, "groundTruth")

        deconstructed, kernel, noise, undersampled_kspace = task.get_deconstructed(ground_truth, get_undersampled_kspace=True)
        print("initial prediction", deconstructed, torch.norm(deconstructed))
        utils.save_image(deconstructed, out_folder, "noisy")
        # undersampled_kspace = y = given measurement
        # deconstructed = F^(-1) y = initial guess TODO: check this

        ###########################################

        prediction = copy.copy(deconstructed)
        var_prediction = Variable(prediction, requires_grad=True) # allows you to calculate gradient with respect to this variable later
        # prediction = x^t = current guess

        final_prediction = methodA(var_prediction, undersampled_kspace, iterations)
        print("predictionA", final_prediction, torch.norm(final_prediction))
        
        utils.save_image(final_prediction.detach(), out_folder, "modelA")

        ###########################################

        prediction = copy.copy(deconstructed)
        var_prediction = Variable(prediction, requires_grad=True) # allows you to calculate gradient with respect to this variable later
        # prediction = x^t = current guess

        final_prediction = methodB(var_prediction, undersampled_kspace, iterations)
        print("predictionB", final_prediction, torch.norm(final_prediction))
        utils.save_image(final_prediction.detach(), out_folder, "modelB")

        ###########################################

        prediction = copy.copy(deconstructed)
        var_prediction = Variable(prediction, requires_grad=True) # allows you to calculate gradient with respect to this variable later
        # prediction = x^t = current guess

        final_prediction = methodC(var_prediction, undersampled_kspace, iterations, net)
        print("predictionC", final_prediction)
        utils.save_image(final_prediction.detach(), out_folder, "modelC")
    break


print("prediction", final_prediction)

    