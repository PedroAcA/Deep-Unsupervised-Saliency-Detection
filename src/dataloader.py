#!/usr/bin/env python

import os
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms.functional import equalize
from config import cfg
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self, config, data_root, data_list, noise_root, pre_trained_model, gt_root=None):
        # Changes in the original code (changes' author: Pedro Aurelio Coelho de Almeida, date: 21st of September, 2021);
        # This function now expects an optional input specifying the path to the ground truth labels dir (e.g.: path/to/MSRA_B/gt/)
        # This function now expects that noise_root to be a string specifying the path to the noisy labels
        # This function now expects an additional required argument specifying whether the model was pre trained (used to normalize inputs)
        self.config = config
        self.sal_root = data_root
        self.sal_source = data_list
        self.noisy_path = noise_root
        self.gt_path = gt_root
        self.pre_trained_model = pre_trained_model
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # Changes in the original code (changes' author: Pedro Aurelio Coelho de Almeida, date: 21st of September, 2021);
        # This function now expects a .lst file with the following format:
        #   img_file.extension  gt_file.extension   noisy_label_1.extension   noisy_label_i.extension OR
        #   img_file.extension  NONE   noisy_label_1.extension   noisy_label_i.extension  when there are no gt labels
        #   Where the noisy labels files are optional
        # This function now expects that all noisy labels (images  provided by noisy classifiers) are in the same dir
        #   Example: all noisy labels corresponding to MSRA-B should be in the MSRA_B/noisy_labels/ dir
        files_list = self.sal_list[item % self.sal_num].split()
        im_name = files_list[0]
        gt_name = files_list[1]

        sal_image = load_image(os.path.join(self.sal_root, im_name), self.pre_trained_model, self.config)

        sal_label = load_sal_label(os.path.join(self.gt_path, gt_name), self.config) if self.gt_path else []# load gt if there is a gt label
        sal_noisy_label = []
        if len(files_list) > 2 and self.noisy_path:  # there are noisy labels
            if(len(files_list)>6):
                print("files_list: {}\n item: {}\n item % self.sal_num: {}".format(files_list,item,item % self.sal_num))

            for noise_file in files_list[2:]:
                sal_noisy_label.append(torch.Tensor(load_image(os.path.join(self.noisy_path, noise_file), self.pre_trained_model, self.config, True)))

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_noisy_label = torch.stack(sal_noisy_label) if sal_noisy_label else torch.tensor([]) # Line modified by Pedro Aurelio Coelho de ALmeida on the 21st of September, 2021 to allow inclusion of empty noisy labels tensor

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'idx': item, 'sal_noisy_label': sal_noisy_label, 'im_name': im_name}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, config, data_root, data_list, gt_root, pre_trained_model):
        # Change in the original code (change's author: Pedro Aurelio Coelho de Almeida, date: 21st of September, 2021);
        # This function now expects a required input specifying the path to the ground truth labels dir (e.g.: path/to/MSRA_B/gt/)
        self.config = config
        self.data_root = data_root
        self.data_list = data_list
        self.gt_root = gt_root
        self.pre_trained_model = pre_trained_model
        with open(self.data_list, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.img_list)

    def __getitem__(self, item):
        # import ipdb; ipdb.set_trace()
        im_name = self.img_list[item % self.image_num].split()[0]
        gt_name = self.img_list[item % self.image_num].split()[1]
        image = load_image(os.path.join(self.data_root, im_name), self.pre_trained_model, self.config)
        image = torch.Tensor(image)
        sal_label = load_sal_label(os.path.join(self.gt_root, gt_name), self.config)
        sal_label = torch.Tensor(sal_label)

        return {'image': image, 'label': sal_label, 'im_name': im_name}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', test_batch_size=1, pin=False):
    # Change in the original code (change's author: Pedro Aurelio Coelho de Almeida, date: 21st of September, 2021);
    # Include changes made in __init__ and __getitem__ of ImageDataTrain
    # Remove unused shuffle in the first line
    drop_last = True
    shuffle = True
    if mode == 'train':
        dataset = ImageDataTrain(config, config.DATA.TRAIN.ROOT, config.DATA.TRAIN.LIST, config.DATA.TRAIN.NOISE_ROOT, config.MODEL.PRE_TRAINED, config.DATA.TRAIN.GT_ROOT)
        batch_size = config.SOLVER.BATCH_SIZE

    if mode == 'val':
        dataset = ImageDataTrain(config, config.DATA.VAL.ROOT, config.DATA.VAL.LIST, config.DATA.VAL.NOISE_ROOT, config.MODEL.PRE_TRAINED, config.DATA.VAL.GT_ROOT)
        batch_size = 1

    if mode == 'test':
        shuffle = False
        drop_last = False
        dataset = ImageDataTest(config, config.DATA.TEST.ROOT, config.DATA.TEST.LIST, config.DATA.TEST.GT_ROOT, config.MODEL.PRE_TRAINED)
        batch_size = test_batch_size

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                  pin_memory=pin, drop_last=drop_last)

    return data_loader

 #Code rewritten by Pedro Aurélio Coelho de Almeida on 21st of September, 2021 to match transformation specified by https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
def load_image(path, pre_trained_model, config, noise=False):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize(config.SOLVER.IMG_SIZE)
    in_ = np.array(im, dtype=np.float32)
    if noise:
        if not len(in_.shape)==2:
            im = im.convert('L') # convert image to grayscale
            in_ = np.array(im, dtype=np.float32)
        if np.max(in_)>1.:
            in_ = in_/255. #put noisy labels in the range [0,1]
        in_ = in_[np.newaxis, ...]
    else:
        if not len(in_.shape)==3:
            im = im.convert('RGB')
            in_ = np.array(im, dtype=np.float32)
        if np.max(in_) > 1:
            in_ = in_ / 255.  # put images in the range [0,1]
        if pre_trained_model:
            in_ = (in_ - np.array((0.485, 0.456, 0.406)))/np.array((0.229, 0.224, 0.225))
        in_ = in_.transpose((2,0,1)) # numpy array of shape (Channel, H, W), instead of (H,W, Channel)
    return in_


 # Code rewritten by Pedro Aurélio Coelho de Almeida on 21st of September, 2021
def load_sal_label(path, config):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize(config.SOLVER.IMG_SIZE)
    label = np.array(im, dtype=np.float32)
    if not len(label.shape) == 2:
        im = im.convert('L') # convert image to grayscale
        label = np.array(im, dtype=np.float32)
    if np.max(label) > 1.:
        label = label / 255.
    label = label[np.newaxis, ...]
    return label

#  #Code excluded by Pedro AUrelio Coelho de ALmeida on the 21st of September, 2021
# def load_noisy_label(path):
#     if not os.path.exists(path):
#         print('File {} not exists'.format(path))
#     im = Image.open(path)
#     im = im.resize(cfg.SOLVER.IMG_SIZE)
#     label = np.array(im, dtype=np.float32)
#     if len(label.shape) == 3:
#         label = label[:,:,0]
#     label = label / 255.
#     label = label[np.newaxis, ...]
#     return label


# def cv_random_flip(img, label):
#     flip_flag = random.randint(0, 1)
#     if flip_flag == 1:
#         img = img[:,:,::-1].copy()
#         label = label[:,:,::-1].copy()
#     return img, label
