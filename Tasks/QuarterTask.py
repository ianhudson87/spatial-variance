import torch
import utils

class Task:
    def __init__(self, stdev_tuple):
        print("Using Quarters task!")
        self.stdev_tuple = stdev_tuple

    def get_deconstructed(self, data):
        if torch.cuda.is_available():
            data = data.cuda()
        h = data.shape[-2]
        w = data.shape[-1]
        img = data
        mid_w = int(w/2)
        mid_h = int(h/2)
        batch_size = data.size()[0]
        
        noise = torch.zeros(img.size())
        for j in range(batch_size):
            # noise for top left quadrant
            noise[j, 0, 0:mid_h, 0:mid_w] = torch.zeros([mid_h, mid_w]).normal_(mean=0, std=self.stdev_tuple[0]/255.)

            # noise for top right quad
            noise[j, 0, 0:mid_h, mid_w:w] = torch.zeros([mid_h, w-mid_w]).normal_(mean=0, std=self.stdev_tuple[1]/255.)

            # noise for bottom left quad
            noise[j, 0, mid_h:h, 0:mid_w] = torch.zeros([h-mid_h, mid_w]).normal_(mean=0, std=self.stdev_tuple[2]/255.)

            # noise for bottom right quad
            noise[j, 0, mid_h:h, mid_w:w] = torch.zeros([mid_h, w-mid_w]).normal_(mean=0, std=self.stdev_tuple[3]/255.)


        noisy_image = img + noise

        utils.imshow(data[3], swap_axes=True)
        utils.imshow(noisy_image[3], swap_axes=True)

        kernel = torch.zeros(batch_size, 15, h, w)
        noise = torch.zeros(batch_size, 1, h, w)
        if torch.cuda.is_available():
            kernel = kernel.cuda()
            noise = noise.cuda()
        return noisy_image, kernel, noise
