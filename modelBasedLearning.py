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
test_name = input("test name:")

data_path = os.path.join("data", "singlecoil_val")
batch_size = 1
mri_image_type = 'reconstruction_rss'
task_name = "undersample"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
# denoiser_model_name = "dncnn"
denoiser_model_name = "dncnn_dynamic_specnorm_more_dyn_layers"
artifact_model_name = "dncnn"
# denoiser_checkpoint_name = "dncnn_constant_noise_1-31-15-47"
# denoiser_checkpoint_name = "dncnn_spec_constant_noise_2-24-16-2"
# denoiser_checkpoint_name = "dncnn_dynamic_specnorm_more_out_layers_vnoise_4-8-20-31"
denoiser_checkpoint_name = "dncnn_dynamic_specnorm_more_dyn_layers_vnoise_4-8-7-13"
artifact_checkpoint_name = "dncnn_Undersample_2-2-5-46"
epoch = 22
out_folder = "modelbased_learning_" + test_name
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
h5_files_train, h5_files_val = utils.train_val_split(h5_files, opt["train_val_split"])

def iterative_method(prediction, given_measurement, denoising_net,
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

def iterative_red(prediction, given_measurement, denoising_net,
                deconstructed, ground_truth,
                measurement_sample_mask, image_num,
                params, verbose=False, return_final_psnr=False):
    # params:
        # lr: learning rate
        # tau: RED weight parameter
        # min_residual = stop after residual is below this value
        # max_iter = maximum number of iterations

    for iteration in range(params["max_iter"]):
        prev_prediction = prediction
        ######################### doing the calculation #########################
        inner_val = measurement_sample_mask * torch.fft.fft(prediction) - given_measurement # Hx - y
        gradient = torch.real(torch.fft.ifft(measurement_sample_mask * inner_val)) # H^T (Hx - y) 

        denoised = torch.clamp(denoising_net(prediction), 0., 1.).detach()
        prediction = prediction - params["lr"] * (gradient + params["tau"] * (prediction - denoised))

        ################# METRICS #######################

        # psnr_measurement = utils.get_psnr(deconstructed, denoised)
        psnr_value = utils.get_psnr(ground_truth, prediction)
        ssim_value = utils.get_ssim(ground_truth, prediction)
        residual = torch.square(torch.norm(prediction - prev_prediction)) / torch.square(torch.norm(prediction))
        objective_fun = torch.norm(torch.absolute(inner_val))

        if iteration % 500 == 0:
            print(image_num, psnr_value)
        writer.add_scalar(f"image_num_{image_num} psnr to GT", psnr_value, iteration)
        writer.add_scalar(f"image_num_{image_num} ssim to GT", ssim_value, iteration)
        writer.add_scalar(f"image_num_{image_num} residual", residual, iteration)
        writer.add_scalar(f"image_num_{image_num} objective function", objective_fun, iteration)
        if verbose: print(f"step {iteration} PSNR: ", psnr_value)

        if residual < params["min_residual"] or iteration > params["max_iter"]:
            print("stop due to residual" if residual < params["min_residual"] else "reached max_iter")
            break

    if return_final_psnr:
        return prediction, psnr_value

    return prediction

deconstructed_psnr_values, deconstructed_ssim_values = [], []
supervised_psnr_values, supervised_ssim_values = [], []
modelbased_final_psnr_values, modelbased_final_ssim_values = [], []
modelbased_max_psnr_values, modelbased_max_ssim_values = [], []

def iterative_red_grid_search(lr_values, tau_values, min_residual_values, max_iter, image_test_nums):
    # returns optimal lr and tau values using RED
    testing_images = []
    best_params = None
    best_psnr = 0
    history = {}

    image_num = -1
    for k in range(len(h5_files_val)):
        data_loader = dataReader.get_dataloader(h5_files_val[k], 'reconstruction_rss', batch_size=batch_size)
        for i, data in enumerate(data_loader):
            image_num += 1
            if image_num in image_test_nums:
                testing_images.append(data)

    for lr in lr_values:
        for tau in tau_values:
            for min_residual in min_residual_values:
                psnr_values = [] # for each testing image, we get a psnr value
                for i, data in enumerate(testing_images):
                    ground_truth = utils.preprocess(data) # ground truth is pixel space
                    writer.add_image(f"gridsearch img {i}", ground_truth[0].cpu())
                    deconstructed, kernel, noise, undersampled_kspace, sample_mask = task.get_deconstructed(ground_truth, seed=image_num, get_undersampled_kspace=True, get_sample_mask=True)
                    prediction = torch.zeros(deconstructed.shape).cuda()
                    iterative_red_params = {
                        "lr": lr,
                        "tau": tau,
                        "min_residual": min_residual,
                        "max_iter": max_iter,
                    }
                    modelbased_prediction, final_psnr = iterative_red(prediction, undersampled_kspace, denoiser_net,
                                                        deconstructed, ground_truth, sample_mask, 0,
                                                        iterative_red_params, return_final_psnr=True)
                    psnr_values.append(final_psnr)
                average_psnr = sum(psnr_values) / len(psnr_values)
                history[tuple([val for val in iterative_red_params.values()])] = average_psnr

                if average_psnr > best_psnr:
                    best_psnr = average_psnr
                    best_params = copy.copy(iterative_red_params)

    return {
        "history": history,
        "best_params": best_params,
        "best_psnr": best_psnr
    }
    
# grid_search_out = iterative_red_grid_search(
#     lr_values = [1, 0.5, 0.25, 0.1, 0.001],
#     tau_values = [1, 0.5, 0.25, 0.1, 0.001],
#     min_residual_values = [1e-9, 1e-10, 1e-11],
#     max_iter = 2000,
#     image_test_num=15)

grid_search_out = iterative_red_grid_search(
    lr_values = [1, 0.5, 0.25, 0.125],
    tau_values = [0.5, 0.25, 0.125, 0.0625],
    min_residual_values = [1e-11],
    max_iter = 2000,
    image_test_nums=[1, 7, 15, 23, 30])
print(grid_search_out)

iterative_red_params = grid_search_out["best_params"]


iterative_params = {
            "lr": 1,
            "denoiser_scale": 1,
            "denoiser_weight": 0.001,
            "min_residual": 1e-10,
            "max_iter": 600,
}

image_num = -1
for k in range(len(h5_files_test)):
    print("k")
    data_loader = dataReader.get_dataloader(h5_files_test[k], 'reconstruction_rss', batch_size=batch_size)

    for i, data in enumerate(data_loader):
        image_num += 1
        ground_truth = utils.preprocess(data) # ground truth is pixel space
        utils.save_image(ground_truth, out_folder, f"{str(image_num)}_groundTruth")

        deconstructed, kernel, noise, undersampled_kspace, sample_mask = task.get_deconstructed(ground_truth, seed=image_num, get_undersampled_kspace=True, get_sample_mask=True)
        # print("deconstructed", deconstructed, torch.norm(deconstructed))
        utils.save_image(deconstructed, out_folder, f"{str(image_num)}_noisy")
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
        utils.save_image(supervised_learning_prediction, out_folder, f"{str(image_num)}_supervised")
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
        # modelbased_prediction = iterative_method(prediction, undersampled_kspace, denoiser_net,
        #                                             deconstructed, ground_truth, sample_mask, image_num,
        #                                             iterative_params)
        modelbased_prediction = iterative_red(prediction, undersampled_kspace, denoiser_net,
                                                    deconstructed, ground_truth, sample_mask, image_num,
                                                    iterative_red_params)
        # print("predictionC", modelbased_prediction)
        utils.save_image(modelbased_prediction.detach(), out_folder, f"{str(image_num)}_modelbased")

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

    