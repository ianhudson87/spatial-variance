import os, glob
import dataReader, utils
import copy
import torch
import torch.linalg
from torch.autograd import Variable
import sampleMask
from tensorboardX import SummaryWriter
import random

random.seed(0)

gpu_num = input("gpu num:")

data_path = os.path.join("data", "singlecoil_val")
iterations = 5000
batch_size = 1
mri_image_type = 'reconstruction_rss'
task_name = "undersample"
learning_rate = 3e-3
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
denoiser_model_name = "dncnn"
artifact_model_name = "dncnn"
denoiser_checkpoint_name = "dncnn_constant_noise_1-31-15-47"
artifact_checkpoint_name = "dncnn_Undersample_2-2-5-46"
epoch = 50
out_folder = "modelbased_learning" + str(1)
out_path = os.path.join("test_logs", out_folder)
writer = SummaryWriter(log_dir=out_path)

# initialize network
denoiser_net = utils.get_model(denoiser_model_name)
artifact_net = utils.get_model(artifact_model_name)
if torch.cuda.is_available():
    denoiser_net.cuda()
    artifact_net.cuda()

# load the checkpoint
checkpoint = torch.load(f'./runs/{checkpoint_name}/net{epoch}.pth')
denoiser_net.load_state_dict(checkpoint['net'])

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

def methodC(var_prediction, given_measurement, iterations, denoising_net, deconstructed, ground_truth, measurement_sample_mask, denoiser_scale=1, denoiser_weight=1):
    # assumes prior is L-2 norm squared
    # var_prediction = x^t
    # given_measurement = y
    # denoising_net = denoiser with pretrained weights
    # deconstructed = pixel space of given measurement ie F^-1(y)
    # ground_truth = pixel space ground truth
    # denoiser_scale = kamilov's method (default=1)
    # denoiser_weight = my method (default=1)
    # measurement_sample_mask = sample mask used on y
    # returns: final prediction (x^(t_final))
    psnr_of_measurement = utils.get_psnr(ground_truth, deconstructed)
    writer.add_image("TEST", torch.fft.fft(torch.absolute(torch.fft.ifft(given_measurement)[0])))
    print("psnr of given measurement to gt: ", psnr_of_measurement)
    writer.add_image(f"given measurement (pixelspace), psnr_to_gt:{psnr_of_measurement}", deconstructed[0].cpu())
    writer.add_image("ground truth (pixelspace)", ground_truth[0].cpu())
    
    for iteration in range(iterations):
        ######################### STEP 1 #########################

        var_prediction_kspace = torch.fft.fft(var_prediction)
        # prediction_kspace = F(x^t)

        # h, w = var_prediction_kspace.shape[-2], var_prediction_kspace.shape[-1]
        # p = opt["sample_percent"]
        # batch_sample_mask = sampleMask.get_batch_sample_mask(h, w, p, 1).cuda()

        var_prediction_kspace_undersampled = var_prediction_kspace * measurement_sample_mask
        # TODO: this should be random mask using same percentage?
        # prediction_kspace_undersampled = MF(x^t) = H(x^t)

        difference = var_prediction_kspace_undersampled - given_measurement
        writer.add_image("given_measurement", given_measurement[0])
        # print("diff", difference)
        # difference = H(x^t) - y

        absolute_difference = torch.abs(difference)
        # take absolute value of each element so that complex values become real values (so that we can take the norm)

        # squared_norm_difference = torch.sum(difference * difference)
        squared_norm_difference = torch.square(torch.norm(absolute_difference, p=2))
        # squared_norm_difference = g(x^t)

        squared_norm_difference.backward() # calculate gradients
        # var_prediction.grad now contains gradient of g(x^t) (wrt x^t)

        # subtract gradient of error wrt prediction from the prediction
        intermediate_prediction = torch.clamp(var_prediction - (learning_rate * var_prediction.grad), 0., 1.)
        print(var_prediction.grad)
        # intermediate_prediction = z^(t+1)

        ######################### STEP 2 #########################

        # Apply denoising to the intermediate prediction
        denoised = torch.clamp((1/denoiser_scale) * denoising_net(intermediate_prediction * denoiser_scale), 0., 1.).detach()
        prediction = denoiser_weight * denoised + (1-denoiser_weight) * intermediate_prediction
        var_prediction = Variable(prediction, requires_grad=True)


        ################# METRICS #######################

        # psnr_measurement = utils.get_psnr(deconstructed, denoised)
        psnr_gt = utils.get_psnr(ground_truth, denoised)
        if iteration % 10 == 0:
            writer.add_image(f"iter:{iteration}, psnr_to_gt:{psnr_gt}", denoised[0].cpu())
        # utils.imshow(denoised[0][0].cpu())
        # This also clears the gradients from var_prediction

        # print(deconstructed)
        # print(denoised)
        # writer.add_scalar("psnr to measurment", psnr_measurement, iteration)
        writer.add_scalar("psnr to GT", psnr_gt, iteration)
        print(f"step {iteration} PSNR_gt: ", psnr_gt)

    return var_prediction

def methodC_grad(var_prediction, given_measurement, iterations, denoising_net, deconstructed, ground_truth, measurement_sample_mask, denoiser_scale=1, denoiser_weight=1, verbose=False):
    # assumes prior is L-2 norm squared
    # var_prediction = x^t
    # given_measurement = y
    # denoising_net = denoiser with pretrained weights
    # deconstructed = pixel space of given measurement ie F^-1(y)
    # ground_truth = pixel space ground truth
    # denoiser_scale = kamilov's method (default=1)
    # denoiser_weight = my method (default=1)
    # measurement_sample_mask = sample mask used on y
    # returns: final prediction (x^(t_final))
    # print(deconstructed)
    # print(var_prediction)
    psnr_of_measurement = utils.get_psnr(ground_truth, deconstructed)
    writer.add_image("TEST", torch.fft.fft(torch.absolute(torch.fft.ifft(given_measurement)[0])))
    print("psnr of given measurement to gt: ", psnr_of_measurement)
    writer.add_image(f"given measurement (pixelspace), psnr_to_gt:{psnr_of_measurement}", deconstructed[0].cpu())
    writer.add_image("ground truth (pixelspace)", ground_truth[0].cpu())
    
    for iteration in range(iterations):
        ######################### STEP 1 #########################
        Ax = measurement_sample_mask * torch.fft.fft(var_prediction) # MFx
        A_trans_Ax = torch.absolute(torch.fft.ifft(measurement_sample_mask * Ax)) # F^{-1}MMFx

        A_trans_y = torch.absolute(torch.fft.ifft(measurement_sample_mask * given_measurement))

        negative_gradient = -A_trans_Ax + A_trans_y

        # distance = measurement_sample_mask * torch.fft.fft(var_prediction) - given_measurement
        # gradient1 = -torch.absolute(torch.fft.ifft(measurement_sample_mask * (distance)))
        # gradient = torch.absolute(torch.fft.ifft(measurement_sample_mask * (distance)))
        if verbose: print("negative_grad", negative_gradient)
        # distance = MFx - y
        # gradient = F^{-1}M (MFx - y) = A^T(Ax - y)
        
        intermediate_prediction = var_prediction + learning_rate * negative_gradient
        if verbose: print("inter", intermediate_prediction)
        # print(torch.clamp(intermediate_prediction, 0., 1.))

        ######################### STEP 2 #########################

        # Apply denoising to the intermediate prediction
        denoised = torch.clamp((1/denoiser_scale) * denoising_net(intermediate_prediction * denoiser_scale), 0., 1.)
        prediction = denoiser_weight * denoised + (1-denoiser_weight) * intermediate_prediction
        var_prediction = Variable(prediction, requires_grad=True)

        if verbose: print("prediction", prediction)
        ################# METRICS #######################

        # psnr_measurement = utils.get_psnr(deconstructed, denoised)
        psnr_gt = utils.get_psnr(ground_truth, prediction.detach())
        if iteration % 500 == 0:
            writer.add_image(f"iter:{iteration}, psnr_to_gt:{psnr_gt}", denoised[0].cpu())
        # utils.imshow(denoised[0][0].cpu())
        # This also clears the gradients from var_prediction

        # print(deconstructed)
        # print(denoised)
        # writer.add_scalar("psnr to measurment", psnr_measurement, iteration)
        writer.add_scalar("psnr to GT", psnr_gt, iteration)
        print(f"step {iteration} PSNR_gt: ", psnr_gt)

    return var_prediction

for k in range(len(h5_files_test)):
    data_loader = dataReader.get_dataloader(h5_files_test[k], 'reconstruction_rss', batch_size=batch_size)

    dncnn_psnr_values = []
    dncnn_ssim_values = []
    modelbased_final_psnr_values = []
    modelbased_max_psnr_values = []
    modelbased_ssim_values = []

    for i, data in enumerate(data_loader):
        if i != 15: continue

        ground_truth = utils.preprocess(data) # ground truth is pixel space
        utils.save_image(ground_truth, out_folder, f"{str(i)}_groundTruth")

        deconstructed, kernel, noise, undersampled_kspace, sample_mask = task.get_deconstructed(ground_truth, get_undersampled_kspace=True, get_sample_mask=True)
        # print("deconstructed", deconstructed, torch.norm(deconstructed))
        utils.save_image(deconstructed, out_folder, f"{str(i)}_noisy")
        # undersampled_kspace = y = given measurement
        # deconstructed = F^(-1) y = initial guess TODO: check this

        ################### MODEL BASED LEARNING #######################

        # prediction = copy.copy(deconstructed)
        prediction = torch.zeros(deconstructed.shape).cuda()
        var_prediction = Variable(prediction, requires_grad=True) # allows you to calculate gradient with respect to this variable later
        # prediction = x^t = current guess

        final_prediction = methodC_grad(var_prediction, undersampled_kspace, iterations, denoiser_net, deconstructed, ground_truth, sample_mask, denoiser_scale=1, denoiser_weight=1e-3)
        # print("predictionC", final_prediction)
        utils.save_image(final_prediction.detach(), out_folder, f"{str(i)}_modelbased")
        print("model_shape", final_prediction.shape)

        #################### SUPERVISED LEARNING (DNCNN) ################
        supervised_learning_prediction = artifact_net(deconstructed)
        print("supervised_shape", supervised_learning_prediction.shape)
    break


print("prediction", final_prediction)

    