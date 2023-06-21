from PIL import Image
from glob import glob
import os
import numpy as np
from torch import tensor
from torchvision.transforms.functional import adjust_gamma

import argparse

''' This module is used to post processes pseudo labels in a folder using gamma correction(gamma=0.5) as described by 
the article.  The output images are saved as 1-channel images using the png format.
'''

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='Path to the input images to be pre-processed', default=None, type=str)
    parser.add_argument('--out_dir', help='Path to save the pre-processed images', default=None, type=str)
    return parser

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
        img = img.convert('L')
        img_tensor = tensor(np.array(img))  # uint8 [H, W]
        img_tensor = adjust_gamma(img_tensor, gamma_factor)
        processed_img = Image.fromarray(img_tensor.numpy())
        processed_img.save(out_dir + img_name + '.png')

if __name__ == "__main__":
    main()

