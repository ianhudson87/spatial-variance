import torch
import random


def get_sample_mask(h, w, p):
    # take columns at random, when a column is chosen, the mirrored column is also automatically chosen
    # h=height, w=width, p=percent of columns chosen
    num_mask_columns = int(w/2*p) # number of columns to choose (not including mirror columns)
    # print(num_mask_columns)
    mask = torch.zeros(h, w)
    columns_chosen = random.sample(range(int((w+1)/2)), num_mask_columns) # columns from left side of image, need to also take the mirror column
    # print(columns_chosen)
    for column in columns_chosen:
        mask[:, column] = 1
        mask[:, (w-1)-column] = 1

    return mask
    # print(mask)
