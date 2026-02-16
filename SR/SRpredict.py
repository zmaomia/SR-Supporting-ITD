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

#from data import myHSTrainingData
#from data import myHSTestData
from data import myHSPredData
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
	test_parser.add_argument("--dataset_name", type=str, default="MS", help="dataset_name, default set to dataset_name")
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

def save_re2img(i, sr_pred, load_dir_LR, save_output_rgb, save_output_5ch, scale_factor):

	
	data_ch = sr_pred.shape[2] 

	file_path, file_name = os.path.split(load_dir_LR[0])

	print('processing image file:', load_dir_LR[0])

	LR_5ch_dataset = gdal.Open(load_dir_LR[0], gdal.GA_ReadOnly)


	width = LR_5ch_dataset.RasterXSize
	height = LR_5ch_dataset.RasterYSize
	geotransform = list(LR_5ch_dataset.GetGeoTransform())
	projection = LR_5ch_dataset.GetProjection()
	bands = LR_5ch_dataset.RasterCount

	# 计算Super-resolution 新的图像尺寸
	new_width = int(width * scale_factor)
	new_height = int(height * scale_factor)

	# 更新GeoTransform以匹配新的分辨率
	geotransform[1] /= scale_factor
	geotransform[5] /= scale_factor

	# 5 bands
	save_5ch_path = os.path.join(save_output_5ch, file_name) 
	driver_5ch = gdal.GetDriverByName('GTiff')
	output_5ch_dataset = driver_5ch.Create(save_5ch_path, new_width, new_height, bands, LR_5ch_dataset.GetRasterBand(1).DataType)
	output_5ch_dataset.SetGeoTransform(geotransform)
	output_5ch_dataset.SetProjection(projection)
	

	# RGB bands
	save_rgb_path = os.path.join(save_output_rgb, file_name)
	driver_rgb = gdal.GetDriverByName('GTiff')
	output_rgb_dataset = driver_rgb.Create(save_rgb_path, new_width, new_height, 3, LR_5ch_dataset.GetRasterBand(1).DataType)
	output_rgb_dataset.SetGeoTransform(geotransform)
	output_rgb_dataset.SetProjection(projection)
	

	# 获取地理变换信息和投影信息
	# input_geotransform = LR_5ch_dataset.GetGeoTransform()
	# input_projection = LR_5ch_dataset.GetProjection()

	#data resolution after up_scaling 
	#x_res = scale_factor *input_geotransform[1]  # Spatial resolution in X  
	#y_res = -scale_factor * input_geotransform[5]	 # Spatial resolution in Y

	#save_rgb_path = os.path.join(save_output_rgb, file_name) 
	#save_5ch_path = os.path.join(save_output_5ch, file_name) 

	# 将 RGB 图像写入新的影像文件
	#driver = gdal.GetDriverByName('GTiff')
	#output_rgb_dataset = driver.Create(save_rgb_path, sr_pred.shape[0], sr_pred.shape[1], 3, gdal.GDT_Byte)

	# 设置输出 RGB 影像的地理参考和投影信息
	#output_rgb_dataset.SetGeoTransform((input_geotransform[0], x_res, input_geotransform[2], 
	#						   input_geotransform[3], input_geotransform[4], y_res))
	#output_rgb_dataset.SetProjection(input_projection)

	#output_5ch_dataset = driver.Create(save_5ch_path, sr_pred.shape[0], sr_pred.shape[1], data_ch, gdal.GDT_Byte)
	# 设置输出 5CH 影像的地理参考和投影信息
	#output_5ch_dataset.SetGeoTransform((input_geotransform[0], x_res, input_geotransform[2], 
	#						   input_geotransform[3], input_geotransform[4], y_res))  #-target_resolution
	#output_5ch_dataset.SetProjection(input_projection)

	for j in range(3):
		output_band_rgb = output_rgb_dataset.GetRasterBand(j + 1)
		output_band_rgb.WriteArray(sr_pred[:, :, j]*255.0)

	for k in range(data_ch):
		output_band_5ch = output_5ch_dataset.GetRasterBand(k + 1)
		output_band_5ch.WriteArray(sr_pred[:, :, k]*255.0)

	# 清理资源
	output_rgb_dataset = None
	output_5ch_dataset = None
	LR_5ch_dataset = None



def test(args):

	test_data_dir = './EXP'
	save_output_rgb = './EXP/RGB_X4'
	save_output_5ch = './EXP/CH5_X4' 

	if not os.path.exists(save_output_rgb):
		os.mkdir(save_output_rgb) 

	if not os.path.exists(save_output_5ch):
		os.mkdir(save_output_5ch) 

	if args.dataset_name=='RGB':
		colors = 3
	elif args.dataset_name=='MS':
		colors = 5


	#for epoch in range (5,165,5):
	model_name = './checkpoints/drone_32_128_X4_5ch/MS_MS_finnalconv_Blocks=9_Feats=128_ckpt_epoch_40.pth'
	#model_name = './checkpoints/' + "Chikusei_Chikusei_finnalconv_Blocks=9_Feats=128_ckpt_epoch_" + str(epoch) +".pth"
	print(model_name)
	ckpt = torch.load(model_name)["model"]

	net = NAFNetSR(args.n_scale, width=args.n_feats, num_blks=args.n_blocks, img_channel=colors, drop_path_rate=args.drop_path_rate, drop_out_rate=args.drop_out_rate, dual=False)
	net.load_state_dict(ckpt)

	net.eval().cuda()
	device = torch.device("cuda" if args.cuda else "cpu")
	print('===> Loading testset')

	test_set = myHSPredData(test_data_dir)
	test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
	scale_factor = args.n_scale # x2, X4, X8
	 
	print('===> Start testing')
	with torch.no_grad():
		output = []
		test_number = 0
		# loading model

		for i, (ms, lms, load_dir_LR) in enumerate(test_loader): # ms:

			ms, lms = ms.to(device), lms.to(device)
			y= net(ms,lms)
			y = y.squeeze().cpu().numpy().transpose(1, 2, 0) 
			#gt = gt.squeeze().cpu().numpy().transpose(1, 2, 0)
			#sr_pred = y[:y.shape[0],:y.shape[1],:] 
			#y = y[:128, :128, :] 
			#save predicted results to images
			save_re2img(i, y, load_dir_LR, save_output_rgb, save_output_5ch, scale_factor)
	  



if __name__ == "__main__":
	main()