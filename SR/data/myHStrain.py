import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import utils
#from skimage.io import imread
import tifffile

def norm_band(img):
    band_max = img.max()
    band_min = img.min()

    return (img-band_min)/(band_max-band_min)

class myHSTrainingData(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        image_dir_HR = os.path.join(image_dir,'HR')
        image_dir_LR = os.path.join(image_dir,'LR')
        image_dir_SR = os.path.join(image_dir,'SR')

        self.image_folders_HR = os.listdir(image_dir_HR)
        self.image_folders_LR = os.listdir(image_dir_LR)
        self.image_folders_SR = os.listdir(image_dir_SR)

        self.image_files_HR = []
        self.image_files_LR = []
        self.image_files_SR = []

        for i in self.image_folders_HR:
            full_path_HR = os.path.join(image_dir_HR, i)
            full_path_LR = os.path.join(image_dir_LR, i)
            full_path_SR = os.path.join(image_dir_SR, i)
            self.image_files_HR.append(full_path_HR)
            self.image_files_LR.append(full_path_LR)
            self.image_files_SR.append(full_path_SR)
        
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    
    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir_HR = self.image_files_HR[file_index]
        load_dir_LR = self.image_files_LR[file_index]
        load_dir_SR = self.image_files_SR[file_index]

        #for tiff image 
        data_HR = tifffile.imread(load_dir_HR)
        data_LR = tifffile.imread(load_dir_LR)
        data_SR = tifffile.imread(load_dir_SR)

        #for .png with 3 channles using skimage.io
        #data_HR = imread(load_dir_HR)
        
        gt = np.array(data_HR, dtype=np.float32)
        ms = np.array(data_LR, dtype=np.float32)
        lms = np.array(data_SR, dtype=np.float32)

        gt = norm_band(gt)
        ms = norm_band(ms)
        lms = norm_band(lms)

        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), utils.data_augmentation(gt, mode=aug_num)

        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        return ms, lms, gt

    def __len__(self):
        return len(self.image_files_HR)*self.factor
