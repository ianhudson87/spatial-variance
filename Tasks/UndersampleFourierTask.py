import sampleMask
import torch
import utils

ifftshift = torch.fft.ifftshift
ifft2 = torch.fft.ifft2
ifftn = torch.fft.ifftn
ifft = torch.fft.ifft
fft = torch.fft.fft
fftshift = torch.fft.fftshift

class UndersampleFourierTask:
    def __init__(self, sample_percent):
        print("Using Undersampling Fourier Space task!")
        self.sample_percent = sample_percent

    def get_deconstructed(self, data):
        if torch.cuda.is_available():
            data = data.cuda()
        h = data.shape[-2]
        w = data.shape[-1]
        batch_size = data.size()[0]
        
        batch_kspace = fft(data) # move to fourier space
        batch_sample_mask = sampleMask.get_batch_sample_mask(h, w, self.sample_percent, batch_size) # generate mask for fourier space
        if torch.cuda.is_available():
            batch_sample_mask = batch_sample_mask.cuda()

        utils.imshow(batch_sample_mask[0][0])

        undersampled_batch_kspace = batch_kspace * batch_sample_mask

        undersampled_batch_image = torch.absolute(ifft(undersampled_batch_kspace))
        kernel = torch.zeros(batch_size, 15, h, w)
        noise = torch.zeros(batch_size, 1, h, w)
        if torch.cuda.is_available():
            kernel = kernel.cuda()
            noise = noise.cuda()
        return undersampled_batch_image, kernel, noise
