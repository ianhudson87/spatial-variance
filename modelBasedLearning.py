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
batch_size = 1
mri_image_type = 'reconstruction_rss'
task_name = "undersample"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
denoiser_model_name = "dncnn"
artifact_model_name = "dncnn"
denoiser_checkpoint_name = "dncnn_constant_noise_1-31-15-47"
artifact_checkpoint_name = "dncnn_Undersample_2-2-5-46"
epoch = 50
out_folder = "modelbased_learning_" + "new"
out_path = os.path.join("test_logs", out_folder)
writer = SummaryWriter(log_dir=out_path)

# initialize network
denoiser_net = utils.get_model(denoiser_model_name)
artifact_net = utils.get_model(artifact_model_name)
if torch.cuda.is_available():
    denoiser_net.cuda()
    artifact_net.cuda()

# load the checkpoints
denoiser_checkpoint = torch.load(f'./runs/{denoiser_checkpoint_name}/net{epoch}.pth')
denoiser_net.load_state_dict(denoiser_checkpoint['net'])

artifact_checkpoint = torch.load(f'./runs/{artifact_checkpoint_name}/net{epoch}.pth')
artifact_net.load_state_dict(artifact_checkpoint['net'])
artifact_net.eval()

# get config about the undersample task
opt = utils.get_options(f"./task_configs/{task_name}_options.json")

task = utils.get_task(task_name, opt, testing=True)

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

def methodC_grad(prediction, given_measurement, denoising_net,
                deconstructed, ground_truth,
                measurement_sample_mask, image_num,
                params, verbose=False):
    # assumes prior is L-2 norm squared
    # prediction = x^t
    # given_measurement = y
    # denoising_net = denoiser with pretrained weights
    # deconstructed = pixel space of given measurement ie F^-1(y)
    # ground_truth = pixel space ground truth
    # measurement_sample_mask = sample mask used on y
    # model parameters:
        # lr: learning rate
        # denoiser_scale = kamilov's method (default=1)
        # denoiser_weight = my method (default=1)
        # min_residual = stop after residual is below this value
        # max_iter = maximum number of iterations

    # returns: final prediction (x^(t_final))

    # print("decon", deconstructed)
    # print(var_prediction)
    psnr_of_measurement = utils.get_psnr(ground_truth, deconstructed)
    # writer.add_image("TEST", torch.fft.fft(torch.absolute(torch.fft.ifft(given_measurement)[0])))
    print("psnr of given measurement to gt: ", psnr_of_measurement)
    # writer.add_image(f"given measurement (pixelspace), psnr_to_gt:{psnr_of_measurement}", deconstructed[0].cpu())
    # writer.add_image("ground truth (pixelspace)", ground_truth[0].cpu())
    

    for iteration in range(params["max_iter"]):
        prev_prediction = prediction
        ######################### STEP 1 #########################
        inner_val = measurement_sample_mask * torch.fft.fft(prediction) - given_measurement # Hx - y
        gradient = torch.real(torch.fft.ifft(measurement_sample_mask * inner_val)) # H^T (Hx - y) 

        # Ax = measurement_sample_mask * torch.fft.fft(var_prediction) # MFx
        # A_trans_Ax = torch.real(torch.fft.ifft(measurement_sample_mask * Ax)) # F^{-1}MMFx

        # A_trans_y = torch.real(torch.fft.ifft(measurement_sample_mask * given_measurement)) #

        # negative_gradient = -A_trans_Ax + A_trans_y

        # distance = measurement_sample_mask * torch.fft.fft(var_prediction) - given_measurement
        # gradient1 = -torch.absolute(torch.fft.ifft(measurement_sample_mask * (distance)))
        # gradient = torch.absolute(torch.fft.ifft(measurement_sample_mask * (distance)))
        if verbose: print("norm of grad", torch.norm(gradient))
        # distance = MFx - y
        # gradient = F^{-1}M (MFx - y) = A^T(Ax - y)
        
        intermediate_prediction = torch.clamp(prediction - params["lr"] * gradient, 0., 1.)
        if verbose: print("inter", intermediate_prediction)
        # print(torch.clamp(intermediate_prediction, 0., 1.))

        prediction = intermediate_prediction
        ######################### STEP 2 #########################

        # Apply denoising to the intermediate prediction
        denoised = torch.clamp((1/params["denoiser_scale"]) * denoising_net(intermediate_prediction * params["denoiser_scale"]), 0., 1.)
        prediction = params["denoiser_weight"] * denoised + (1-params["denoiser_weight"]) * intermediate_prediction
        prediction = prediction.detach()

        ################# METRICS #######################

        # psnr_measurement = utils.get_psnr(deconstructed, denoised)
        psnr_value = utils.get_psnr(ground_truth, prediction)
        ssim_value = utils.get_ssim(ground_truth, prediction)
        residual = torch.square(torch.norm(prediction - prev_prediction)) / torch.square(torch.norm(prediction))
        objective_fun = torch.norm(torch.absolute(inner_val))

        if iteration % 50 == 0:
            print(image_num, psnr_value)
        #     writer.add_image(f"iter:{iteration}, psnr_to_gt:{psnr_val}", denoised[0].cpu())
        # utils.imshow(denoised[0][0].cpu())
        # This also clears the gradients from var_prediction

        # print(deconstructed)
        # print(denoised)
        # writer.add_scalar("psnr to measurment", psnr_measurement, iteration)
        writer.add_scalar(f"image_num_{image_num} psnr to GT", psnr_value, iteration)
        writer.add_scalar(f"image_num_{image_num} ssim to GT", ssim_value, iteration)
        writer.add_scalar(f"image_num_{image_num} residual", residual, iteration)
        writer.add_scalar(f"image_num_{image_num} objective function", objective_fun, iteration)
        if verbose: print(f"step {iteration} PSNR: ", psnr_value)

        if residual < params["min_residual"] or iteration > params["max_iter"]:
            print("stop due to residual" if residual < params["min_residual"] else "reached max_iter")
            break

    return prediction



image_num=0
deconstructed_psnr_values, deconstructed_ssim_values = [], []
supervised_psnr_values, supervised_ssim_values = [], []
modelbased_final_psnr_values, modelbased_final_ssim_values = [], []
modelbased_max_psnr_values, modelbased_max_ssim_values = [], []

for k in range(len(h5_files_test)):
    data_loader = dataReader.get_dataloader(h5_files_test[k], 'reconstruction_rss', batch_size=batch_size)

    for i, data in enumerate(data_loader):
        image_num += 1

        ground_truth = utils.preprocess(data) # ground truth is pixel space
        utils.save_image(ground_truth, out_folder, f"{str(i)}_groundTruth")

        deconstructed, kernel, noise, undersampled_kspace, sample_mask = task.get_deconstructed(ground_truth, seed=image_num, get_undersampled_kspace=True, get_sample_mask=True)
        # print("deconstructed", deconstructed, torch.norm(deconstructed))
        utils.save_image(deconstructed, out_folder, f"{str(i)}_noisy")
        # undersampled_kspace = y = given measurement
        # deconstructed = F^(-1) y = initial guess TODO: check this
        deconstructed_psnr, deconstructed_ssim = utils.get_psnr(ground_truth, deconstructed), utils.get_ssim(ground_truth, deconstructed)
        deconstructed_psnr_values.append(deconstructed_psnr)
        deconstructed_ssim_values.append(deconstructed_ssim)

        #################### SUPERVISED LEARNING (DNCNN) ################
        supervised_learning_prediction = torch.clamp(artifact_net(deconstructed), 0., 1.).detach()
        # for p in artifact_net.parameters():
        #         print (p.data[0][0][0][0])
        #         break
        utils.save_image(supervised_learning_prediction, out_folder, f"{str(i)}_supervised")
        # print("supervised_shape", supervised_learning_prediction.shape)
        supervised_psnr, supervised_ssim = utils.get_psnr(ground_truth, supervised_learning_prediction), utils.get_ssim(ground_truth, supervised_learning_prediction)
        supervised_psnr_values.append(supervised_psnr)
        supervised_ssim_values.append(supervised_ssim)
        ####################################################################
        # break

        ################### MODEL BASED LEARNING #######################

        # prediction = copy.copy(deconstructed)
        prediction = torch.zeros(deconstructed.shape).cuda()
        # var_prediction = Variable(prediction, requires_grad=True) # allows you to calculate gradient with respect to this variable later
        # prediction = x^t = current guess
        params = {
            "lr": ,
            "denoiser_scale": 1,
            "denoiser_weight": 1,
            "min_residual": 1e-9,
            "max_iter": 3000,
        }
        modelbased_prediction = methodC_grad(prediction, undersampled_kspace, denoiser_net,
                                                    deconstructed, ground_truth, sample_mask, image_num,
                                                    params)
        # print("predictionC", modelbased_prediction)
        utils.save_image(modelbased_prediction.detach(), out_folder, f"{str(i)}_modelbased")

        # print("model_shape", modelbased_prediction.shape)

        model_based_final_psnr, model_based_final_ssim = utils.get_psnr(ground_truth, modelbased_prediction), utils.get_ssim(ground_truth, modelbased_prediction)
        modelbased_final_psnr_values.append(model_based_final_psnr)
        modelbased_final_ssim_values.append(model_based_final_ssim)
        ################################################################
        # break
    # break

utils.write_test_file(deconstructed_psnr_values, deconstructed_ssim_values, out_folder, "deconstructed")
utils.write_test_file(supervised_psnr_values, supervised_ssim_values, out_folder, "supervised")
utils.write_test_file(modelbased_final_psnr_values, modelbased_final_ssim_values, out_folder, "modelbased_final")

# print("prediction", final_prediction)

    