from PIL import Image
from glob import glob
import os
import numpy as np
from torch import tensor
from torchvision.transforms.functional import adjust_gamma

data_root = 'pseudo_labels_pre_processed_2048_2048/'
ext = '.png'
files_in_data_root = list(sorted(glob(data_root + "*" + ext)))
post_processed_dir = 'post_processed_' + data_root
os.makedirs(post_processed_dir, exist_ok=True)
gamma_factor = 0.5

for f in files_in_data_root:
    print("Pre processing {}".format(f))
    img_name = os.path.basename(f)[:-4]  # removes extension from filename
    img = Image.open(f)
    img = img.convert('L')
    img_tensor = tensor(np.array(img))# uint8 [H, W]
    img_tensor = adjust_gamma(img_tensor, gamma_factor)
    processed_img = Image.fromarray(img_tensor.numpy())
    processed_img.save(post_processed_dir + img_name + ext)