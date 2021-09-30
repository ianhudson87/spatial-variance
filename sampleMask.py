import torch
import random


def get_sample_mask(h, w, p):
    # take columns at random
    # h=height, w=width, p=percent of columns chosen
    num_mask_columns = int(w*p) # number of columns to choose
    # print(num_mask_columns)
    mask = torch.zeros(h, w)
    columns_chosen = random.sample(range(w), num_mask_columns) # columns from left side of image, need to also take the mirror column
    # print(columns_chosen)
    for column in columns_chosen:
        mask[:, column] = 1

    return mask

def get_batch_sample_mask(h, w, p, batch_size):
    # take columns at random
    # h=height, w=width, p=percent of columns chosen, batch_size
    num_mask_columns = int(w*p) # number of columns to choose
    mask = torch.zeros(batch_size, 1, h, w)

    for batch in range(batch_size):
        columns_chosen = random.sample(range(w), num_mask_columns) # columns from left side of image, need to also take the mirror column
        # print(columns_chosen)
        for column in columns_chosen:
            mask[batch, 0, :, column] = 1

    return mask
