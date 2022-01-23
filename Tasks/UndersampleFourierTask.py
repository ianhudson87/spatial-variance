import sampleMask
import torch, torch.fft
import utils
import random

# ifftshift = torch.fft.ifftshift
# ifft2 = torch.fft.ifft2
# ifftn = torch.fft.ifftn
ifft = torch.fft.ifft
fft = torch.fft.fft
# fftshift = torch.fft.fftshift

class Task:
    def __init__(self, sample_percent, testing=False):
        print("Using Undersampling Fourier Space task!", "sample percent", sample_percent, "testing", testing)
        self.sample_percent = sample_percent
        self.testing = testing

    def get_deconstructed(self, data, seed=None):
        if torch.cuda.is_available():
            data = data.cuda()
        h = data.shape[-2]
        w = data.shape[-1]
        batch_size = data.size()[0]
        
        batch_kspace = fft(data) # move to fourier space
        
        
        if self.testing:
            if seed is None:
                assert ValueError("testing needs seed per datapoint")
            else:
                random.seed(seed)
        batch_sample_mask = sampleMask.get_batch_sample_mask(h, w, self.sample_percent, batch_size) # generate mask for fourier space
        if torch.cuda.is_available():
            batch_sample_mask = batch_sample_mask.cuda()

        # utils.imshow(batch_sample_mask[0][0])

        undersampled_batch_kspace = batch_kspace * batch_sample_mask

        undersampled_batch_image = torch.absolute(ifft(undersampled_batch_kspace))
        kernel = torch.zeros(batch_size, 15, h, w)
        noise = torch.zeros(batch_size, 1, h, w)
        if torch.cuda.is_available():
            kernel = kernel.cuda()
            noise = noise.cuda()
            
        undersampled_batch_image = torch.clamp(undersampled_batch_image, 0., 1.)
        return undersampled_batch_image, kernel, noise
