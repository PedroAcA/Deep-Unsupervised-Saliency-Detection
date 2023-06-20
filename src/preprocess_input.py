from PIL import Image, ImageSequence
from glob import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import cv2
from torch import tensor, histc, cumsum, round
import torchvision
from torchvision.transforms.functional import adjust_gamma, resize
from torchvision.transforms import InterpolationMode
import argparse

''' This module pre processes images in a folder using histogram equalization followed by gamma correction(gamma=0.5) 
 as described by the article.  The output images are saved as 3-channel images using the JPG format.
'''

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='Path to the input images to be pre-processed', default=None, type=str)
    parser.add_argument('--out_dir', help='Path to save the pre-processed images', default=None, type=str)
    return parser


def chan_equalize(in_img):
    '''
    Performs histogram equalization for signle channel image according to the method described by Gozanlez et al. 's
    Digital Image Processing book.

        Parameters:
            in_img: a single channel image (can be a grayscale image or one of the three RGB color images for examples)
            stored as a CPU pytorch tensor. Its values can range from 0 to 255

       Returns:
           A single channel image (can be a grayscale image or one of the three RGB color images for examples) stored
           as a CPU pytorch tensor corresponding to the histogram equalized in_img. Its values can range from 0 to 255
    '''
    L = 256 # levels from 0 to 255
    float_img= in_img.detach().clone().to(torch.float)
    hist = histc(float_img, bins=L, min=0, max=L-1)
    cdf = cumsum(hist, dim=0)
    num_pixels = in_img.shape[-2] * in_img.shape[-1]
    t_rk = round((L-1)*cdf/num_pixels).to(torch.uint8)
    return t_rk[float_img.to(torch.int64)]


def equalize(in_img):
    '''
       Performs histogram equalization for signle or multi channel image according to the method described by
       Gozanlez et al. 's Digital Image Processing book.

           Parameters:
               in_img: a single or multi channel image stored as a CPU pytorch tensor. Its values can range from 0 to 255

          Returns:
              A single or multi channel image stored as a CPU pytorch tensor corresponding to the histogram equalized in_img.
              Its values can range from 0 to 255
       '''
    return torch.stack([chan_equalize(in_img[c]) for c in range(in_img.size(0))])

def main():
    parser = create_parser()
    args = parser.parse_args()
    data_root = args.data_root
    files_in_data_root = list(sorted(glob(data_root + "*")))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    gamma_factor = 0.5
    for f in files_in_data_root:
        print("Pre processing {}".format(f))
        img_name = os.path.basename(f)[:-4]  # removes extension from filename
        img = Image.open(f)
        img = img.convert('RGB')
        img_tensor = tensor(np.array(img).transpose((2, 0, 1)))# uint8 [C, H, W]
        img_tensor[img_tensor == 0] = 255
        img_tensor = equalize(img_tensor)
        img_tensor = adjust_gamma(img_tensor, gamma_factor)
        processed_img = Image.fromarray(img_tensor.permute(1, 2, 0).numpy())
        processed_img.save(out_dir + img_name + '.jpg')


if __name__ == "__main__":
    main()



