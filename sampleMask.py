import torch
import random


# def get_sample_mask(h, w, p):
#     if (p<0.2):
#         print("bad sample perecentage, too low")
#         return None
#     # take columns at random
#     # h=height, w=width, p=percent of columns chosen
#     num_mask_columns = int(w*(p-0.2)) # number of columns to choose
#     # print(num_mask_columns)
#     mask = torch.zeros(h, w)
#     columns_chosen = random.sample(range(w), num_mask_columns) # columns from left side of image, need to also take the mirror column
#     # print(columns_chosen)
#     for column in columns_chosen:
#         mask[:, column] = 1

#     return mask

def get_batch_sample_mask(h, w, p, batch_size):
    if (p<0.2):
        print("bad sample perecentage, too low")
        return None
    # choose first 10 percent and last 10 perecent of columsn, then take columns at random
    num_mask_columns = int(w*(p-0.2)) # number of columns to choose
    mask = torch.zeros(batch_size, 1, h, w)

    for batch in range(batch_size):
        columns_chosen = list(range(0, int(w*0.1)+1))
        columns_chosen += list(range(int(w*0.9), w))
        columns_chosen += random.sample(range(int(w*0.1)+1, int(w*0.9)), num_mask_columns)
        # print(columns_chosen)
        for column in columns_chosen:
            mask[batch, 0, :, column] = 1

    return mask

def sample_image(image_batch, p):
    h = image_batch.shape[-2]
    w = image_batch.shape[-1]
    batch_size = image_batch.shape[0]
    batch_sample_mask = get_batch_sample_mask(h, w, p, batch_size)

    if torch.cuda.is_available():
        batch_sample_mask = batch_sample_mask.cuda()

    return image_batch * batch_sample_mask
