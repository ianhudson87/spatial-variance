import torch
import math
import random

class VariableNoiseTask:
    def __init__(self, noise_min, noise_max, patch_size):
        print("Using Variable Noise Task!")
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.patch_size = patch_size
    
    def get_deconstructed(self, data):
        if torch.cuda.is_available():
            data = data.cuda()
        h = data.shape[-2]
        w = data.shape[-1]
        img = data
        patch_size = self.patch_size
        noise_max = self.noise_max
        noise_min = self.noise_min
        batch_size = data.size()[0]

        noise = torch.zeros(img.size())
        for j in range(batch_size):
            for row in range(math.ceil(h/patch_size)):
                for col in range(math.ceil(w/patch_size)):
                    # determine last index for row
                    if patch_size*(row+1) > h:
                        row_end_point = h
                    else:
                        row_end_point = patch_size*(row+1)
                    # determine last index for col
                    if patch_size*(col+1) > w:
                        col_end_point = w
                    else:
                        col_end_point = patch_size*(col+1)
                    val_noiseL = random.random()*(noise_max - noise_min) + noise_min
                    # print(noise[j,0, opt.patch_size*row:row_end_point, opt.patch_size*col:col_end_point])
                    # print(torch.FloatTensor([row_end_point-opt.patch_size*row, col_end_point-opt.patch_size*col]).normal_(mean=0, std=noiseL))
                    noise[j, 0, patch_size*row:row_end_point, patch_size*col:col_end_point] = torch.zeros([row_end_point-patch_size*row, col_end_point-patch_size*col]).normal_(mean=0, std=val_noiseL/255.)
        
        kernel = torch.zeros(batch_size, 15, h, w)
        if torch.cuda.is_available():
            noise = noise.cuda()
            kernel = kernel.cuda()
        noisy_image = img + noise
        return noisy_image, kernel, noise