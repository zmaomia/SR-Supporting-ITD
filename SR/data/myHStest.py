import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch


class myHSTestData(data.Dataset):
    def __init__(self, image_dir, use_3D=False):
        test_data = sio.loadmat(image_dir)
        self.use_3Dconv = use_3D
        
        self.ms = np.array(test_data['ms'][...], dtype=np.float32)
        self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
        self.gt = np.array(test_data['gt'][...], dtype=np.float32)

        self.msA = np.array(test_data['msA'][...], dtype=np.float32)
        self.lmsA = np.array(test_data['ms_bicubicA'][...], dtype=np.float32)
        self.gtA = np.array(test_data['gtA'][...], dtype=np.float32)

    def __getitem__(self, index):
        
        gt = self.gt[index, :, :, :]
        ms = self.ms[index, :, :, :]
        lms = self.lms[index, :, :, :]

        gtA = self.gtA[index, :, :, :]
        msA = self.msA[index, :, :, :]
        lmsA = self.lmsA[index, :, :, :]

        if self.use_3Dconv:
            ms, lms, gt, msA, lmsA, gtA = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :], msA[np.newaxis, :, :, :], lmsA[np.newaxis, :, :, :], gtA[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)

            msaA = torch.from_numpy(msaA.copy()).permute(0, 3, 1, 2)
            lmsA = torch.from_numpy(lmsA.copy()).permute(0, 3, 1, 2)
            gtA = torch.from_numpy(gtA.copy()).permute(0, 3, 1, 2)
        
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

            msA = torch.from_numpy(msA.copy()).permute(2, 0, 1)
            lmsA = torch.from_numpy(lmsA.copy()).permute(2, 0, 1)
            gtA = torch.from_numpy(gtA.copy()).permute(2, 0, 1)
        
        return ms, lms, gt, msA, lmsA, gtA

    def __len__(self):
        return self.gt.shape[0]
