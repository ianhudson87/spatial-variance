import sampleMask
import torch

ifftshift = torch.fft.ifftshift
ifft2 = torch.fft.ifft2
ifftn = torch.fft.ifftn
ifft = torch.fft.ifft
fft = torch.fft.fft
fftshift = torch.fft.fftshift

class UndersampleFourierTask:
    def __init__(self, sample_percent, batch_size):
        print("Using Undersampling Fourier Space task!")
        self.sample_percent = sample_percent
        self.batch_size = batch_size

    def get_deconstructed(self, data):
        batch_kspace = fft(data) # move to fourier space
        h = batch_kspace.shape[-2]
        w = batch_kspace.shape[-1]
        batch_sample_mask = sampleMask.get_batch_sample_mask(h, w, self.sample_percent, self.batch_size) # generate mask for fourier space
        
        undersampled_batch_kspace = batch_kspace * batch_sample_mask

        undersampled_batch_image = torch.absolute(ifft(undersampled_batch_kspace))
        inputs = undersampled_batch_image
        kernel = torch.zeros(self.batch_size, 15, h, w)
        noise = torch.zeros(self.batch_size, 1, h, w)
        print("HERE", inputs.shape, kernel.shape, noise.shape)
        return inputs, kernel, noise