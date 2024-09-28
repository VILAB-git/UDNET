import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import cv2
from pathlib import Path
import yaml
import util.util as util
import albumentations

import random

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'dsec_train', 'day')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'dsec_train', 'night')  # create a path '/path/to/data/trainB'
        
        if opt.phase == "test" :
            self.dir_A = os.path.join(opt.dataroot, 'dsec_val', "day")
            self.dir_B = os.path.join(opt.dataroot, 'dsec_val', "night")
        
        self.A_paths = make_dataset(self.dir_A, opt.max_dataset_size)   # load images from '/path/to/data/trainA'
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.dir_A_no_rectify = self.dir_A.replace("bin_3","bin_3_no_rectify")
        self.dir_B_no_rectify = self.dir_B.replace("bin_3","bin_3_no_rectify")
        
        self.A_no_rectify_paths = make_dataset(self.dir_A_no_rectify, opt.max_dataset_size)
        self.B_no_rectify_paths = make_dataset(self.dir_B_no_rectify, opt.max_dataset_size)
        # random.shuffle(self.A_no_rectify_paths)
        # random.shuffle(self.B_no_rectify_paths)
        # import pdb; pdb.set_trace()
        self.A_no_rectify_size = len(self.A_no_rectify_paths)  # get the size of dataset A
        self.B_no_rectify_size = len(self.B_no_rectify_paths)  # get the size of dataset B
        
        assert self.A_size == self.A_no_rectify_size
        assert self.B_size == self.B_no_rectify_size
        
        self.size = opt.crop_size
        
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if opt.phase == 'train':
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if index % 2 == 0:
            idx = int(index/2)
            A_path = self.A_paths[idx % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = idx % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        else:
            idx = int((index-1)/2)
            A_path = self.A_no_rectify_paths[idx % self.A_no_rectify_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = idx % self.B_no_rectify_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_no_rectify_size - 1)
            B_path = self.B_no_rectify_paths[index_B]
        
        A_img = np.load(A_path)
        B_img = np.load(B_path)
        
        # A_img = np.stack([A_img[0], A_img[1], np.zeros_like(A_img[0])], axis=0)
        # B_img = np.stack([B_img[0], B_img[1], np.zeros_like(B_img[0])], axis=0)

        # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt)
        
        A_img = self.preprocessor(image=A_img.astype(np.float32).transpose(1,2,0))["image"]
        B_img = self.preprocessor(image=B_img.astype(np.float32).transpose(1,2,0))["image"]
        
        # A_img = (A_img / np.max(A_img))
        # B_img = (B_img / np.max(B_img)) 
        
        A_img = A_img / (np.max(A_img)/2.0) - 1.0
        B_img = B_img / (np.max(B_img)/2.0) - 1.0
        
        A = torch.from_numpy(A_img).permute(2,0,1)
        B = torch.from_numpy(B_img).permute(2,0,1)
        
        if index % 2 == 1:
            # import pdb; pdb.set_trace()
            A_path = A_path.replace('.npy','no_rectify.npy')
            B_path = B_path.replace('.npy','no_rectify.npy')
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size) * 2
