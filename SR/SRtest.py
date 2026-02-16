import argparse
import os
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.nn as nn
import torch

import scipy.io as sio
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json

from data import myHSTrainingData
from data import myHSTestData
from NAFSR import NAFNetSR
from NAFcommon import *

# loss
from loss import reconstruction_SADloss
from loss import HybridLoss
from loss import CharbonnierLoss
# from loss import HyLapLoss
from metrics import quality_assessment

from osgeo import gdal

# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = './data/SR/'


def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    test_parser = subparsers.add_parser("test", help="parser for training arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 64")
    test_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
    test_parser.add_argument("--n_feats", type=int, default=128, help="n_feats, default set to 256")
    test_parser.add_argument("--n_blocks", type=int, default=9, help="n_blocks, default set to 6")
    test_parser.add_argument("--drop_path_rate", type=float, default=0.1, help="drop_path_rate, default set to 0.1")
    test_parser.add_argument("--drop_out_rate", type=float, default=0, help="drop_out_rate, default set to 0.1")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    test_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    test_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--model_title", type=str, default="UnSR", help="model_title, default set to model_title")
    test_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    test_parser.add_argument("--learning_rate", type=float, default=5e-3,
                              help="learning rate, default set to 1e-4")
    test_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    test_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    test_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    # test_parser.add_argument("--test_dir", type=str, required=True, help="directory of testset")
    # test_parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    test(args)

def save_re2img(i, sr_pred):

    data_ch = sr_pred.shape[2]

    save_rgb_path = './Prediction/' + str(i) + '_rgb.tif'
    save_5ch_path = './test/' + str(i) + '_5ch.tif'


    driver = gdal.GetDriverByName('GTiff')
    output_rgb_dataset = driver.Create(save_rgb_path, sr_pred.shape[0], sr_pred.shape[1], 3, gdal.GDT_Byte)
    #output_rgb_dataset.SetGeoTransform(geotransform)
    #output_rgb_dataset.SetProjection(projection)

    output_5ch_dataset = driver.Create(save_5ch_path, sr_pred.shape[0], sr_pred.shape[1], data_ch, gdal.GDT_Byte)

    #output_5ch_dataset.SetGeoTransform((input_geotransform[0], target_resolution, input_geotransform[2], 
                               #input_geotransform[3], input_geotransform[4], -target_resolution))
    #output_5ch_dataset.SetProjection(input_projection)

    for i in range(3):
        output_band_rgb = output_rgb_dataset.GetRasterBand(i + 1)
        output_band_rgb.WriteArray(sr_pred[:, :, i]*255.0)

    for i in range(data_ch):
        output_band_5ch = output_5ch_dataset.GetRasterBand(i + 1)
        output_band_5ch.WriteArray(sr_pred[:, :, i]*255.0)

    output_rgb_dataset = None
    output_5ch_dataset = None



def test(args):

    #train_path    = '/data/yuy/SSPSR-master/undataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/trains/'
    eval_path     = '/Prediction/SR/'
    result_path   = './Prediction/'
    #test_data_dir = '/data/yuy/SSPSR-master/undataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/'+args.dataset_name+'_test.mat'
    
    #train_set = myHSTrainingData(image_dir=train_path, augment=True)
    eval_set = myHSTrainingData(image_dir=eval_path, augment=False)
    #train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)


    if args.dataset_name=='RGB':
        colors = 3
    elif args.dataset_name=='MS':
        colors = 5
    #else:
        #colors = 7 

    #for epoch in range (5,165,5):
    model_name = './checkpoints/drone_32_64_X2_5ch/MS_MS_finnalconv_Blocks=9_Feats=128_ckpt_epoch_40.pth'
    #model_name = './checkpoints/' + "Chikusei_Chikusei_finnalconv_Blocks=9_Feats=128_ckpt_epoch_" + str(epoch) +".pth"
    print(model_name)
    ckpt = torch.load(model_name)["model"]

    net = NAFNetSR(up_scale=args.n_scale, width=args.n_feats, num_blks=args.n_blocks, img_channel=colors, drop_path_rate=args.drop_path_rate, drop_out_rate=args.drop_out_rate, dual=False)
    net.load_state_dict(ckpt)

    net.eval().cuda()
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    #test_set = myHSTestData(test_data_dir)
    test_set = myHSTrainingData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():
        output = []
        test_number = 0
        # loading model

        for i, (ms, lms, gt) in enumerate(test_loader):
            #print(test_set[i])
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
        #for i, (ms, lms, gt, msA, lmsA, gtA) in enumerate(test_loader):
            #ms, lms, gt, msA, lmsA, gtA = ms.to(device), lms.to(device), gt.to(device), msA.to(device), lmsA.to(device), gtA.to(device)
            y= net(ms,lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:] 

            #save predicted results to images
            #save_re2img(i, y)
            # output evaluation results

            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        
        for index in indices:
            indices[index] = indices[index] / test_number

    save_dir = "/Prediction/SR.npy"
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)
    QIstr = './PredictionSR.txt'
    json.dump(indices, open(QIstr, 'w'))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

if __name__ == "__main__":
    main()