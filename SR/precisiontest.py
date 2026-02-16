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
from unsr import UnSR
from unsrcommon import *

# loss
from loss import reconstruction_SADloss
from loss import HybridLoss
from loss import CharbonnierLoss
# from loss import HyLapLoss
from metrics import quality_assessment

# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = ''




def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    test_parser = subparsers.add_parser("test", help="parser for training arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 64")
    test_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
    test_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    test_parser.add_argument("--n_blocks", type=int, default=9, help="n_blocks, default set to 6")
    test_parser.add_argument("--num_dense_layer", type=int, default=6, help="num_dense_layer, default set to 4")
    test_parser.add_argument("--growth_rate", type=int, default=32, help="growth_rate, default set to 16")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    test_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    test_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--model_title", type=str, default="UnSR", help="model_title, default set to model_title")
    test_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    test_parser.add_argument("--learning_rate", type=float, default=1e-4,
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

def test(args):

    train_path    = './undataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/trains/'
    eval_path     = './undataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/evals/'
    result_path   = './undataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/tests/'
    test_data_dir = './undataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/'+args.dataset_name+'_test.mat'
    
    train_set = myHSTrainingData(image_dir=train_path, augment=True)
    eval_set = myHSTrainingData(image_dir=eval_path, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    else:
        colors = 128    

    for epoch in range (5,165,5):
        model_name = './checkpoints/' + "Chikusei_Chikusei_44641280.520UnSR_Blocks=4_Feats=128_ckpt_epoch_" + str(epoch) +".pth"
        print(model_name)
        ckpt = torch.load(model_name)["model"]

        net = UnSR(n_colors=128, n_feats=128, num_dense_layer = 4,growth_rate = 64,n_blocks = 4,up_scale = 4)
        net.load_state_dict(ckpt)

        net.eval().cuda()
        device = torch.device("cuda" if args.cuda else "cpu")
        print('===> Loading testset')
        test_set = myHSTestData(test_data_dir)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        print('===> Start testing')
        with torch.no_grad():
            output = []
            test_number = 0
            # loading model

            for i, (ms, lms, gt, msA, lmsA, gtA) in enumerate(test_loader):
                ms, lms, gt, msA, lmsA, gtA = ms.to(device), lms.to(device), gt.to(device), msA.to(device), lmsA.to(device), gtA.to(device)
                y= net(ms,lms)
                y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
                y = y[:gt.shape[0],:gt.shape[1],:] 
                if i==0:
                    indices = quality_assessment(gt, y, data_range=1., ratio=4)
                else:
                    indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
                output.append(y)
                test_number += 1
            for index in indices:
                indices[index] = indices[index] / test_number

        # save_dir = "/data/test.npy"
        save_dir = result_path + 'test' + '.npy'
        np.save(save_dir, output)
        print("Test finished, test results saved to .npy file at ", save_dir)
        print(indices)
        QIstr = 'test'+'_'+str(time.ctime())+ ".txt"
        json.dump(indices, open(QIstr, 'w'))



def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

if __name__ == "__main__":
    main()