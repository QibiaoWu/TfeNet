import os
import time
import numpy as np
import data_BAS as data
from importlib import import_module
import shutil
from trainval_classifier_BAS import train_casenet, val_casenet
from utils import Logger, save_itk, weights_init
import sys
sys.path.append('../')
from split_combine_mj import SplitComb
import torch
from torch.nn import DataParallel
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import csv
from option import parser
import gc
import warnings

warnings.filterwarnings("ignore")

def main():
	global args
	args = parser.parse_args()
	torch.manual_seed(0)
	torch.cuda.set_device(args.device)
	print('----------------------Load Model------------------------')
	model = import_module(args.model)
	config, net = model.get_model(args)
	start_epoch = args.start_epoch
	save_dir = args.save_dir
	save_dir = os.path.join('results',save_dir)
	print("savedir: ", save_dir)
	print("args.lr: ", args.lr)
	args.lr_stage = config['lr_stage']
	args.lr_preset = config['lr']
	os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
	
	if args.resume:
		resume_part = args.resumepart
		checkpoint = torch.load(args.resume)

		if resume_part:
			"""
			load part of the weight parameters
			"""
			net.load_state_dict(checkpoint['state_dict'], strict=False)
			print('part load Done')
		else:
			"""
			load full weight parameters
			"""
			net.load_state_dict(checkpoint['state_dict'])
			print("full resume Done")
	else:
		weights_init(net, init_type='xavier')  # weight initialization

	if args.epochs is None:
		end_epoch = args.lr_stage[-1]
	else:
		end_epoch = args.epochs
		
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	logfile = os.path.join(save_dir, 'log.txt')
	
	if args.test != 1:
		sys.stdout = Logger(logfile)
		pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
		for f in pyfiles:
			shutil.copy(f, os.path.join(save_dir, f))
			
	if torch.cuda.is_available() and args.multigpu != 1:
		# net = torch.nn.DataParallel(net).cuda()
		net = net.cuda()
	cudnn.benchmark = True

	if args.multigpu:
		net = DataParallel(net).cuda()

	if args.cubesizev is not None:
		marginv = args.cubesizev
	else:
		marginv = args.cubesize
	print('validation stride ', args.stridev)

	if not args.sgd:
		optimizer = optim.Adam(net.parameters(), lr=2e-2)  # args.lr
		# optimizer = optim.AdamW(net.parameters(), lr=0.01)
	else:
		optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

	if args.test:
		print('---------------------testing---------------------')
		split_comber = SplitComb(args.stridev, marginv)
		dataset_test = data.AirwayData(
			config,
			phase='test',
			split_comber=split_comber,
			debug=args.debug,
			random_select=False)
		test_loader = DataLoader(
			dataset_test,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.workers,
			pin_memory=True)
		epoch = 1
		print('start testing')
		testdata = val_casenet(epoch, net, test_loader, args, save_dir, test_flag=True)
		return

	if args.debugval:
		epoch = 1
		print ('---------------------validation---------------------')
		split_comber = SplitComb(args.stridev, marginv)
		dataset_val = data.AirwayData(
			config,
			phase='val',
			split_comber=split_comber,
			debug=args.debug,
			random_select=False)
		val_loader = DataLoader(
			dataset_val,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.workers,
			pin_memory=True)
		valdata = val_casenet(epoch, net, val_loader, args, save_dir)
		return

	print('---------------------------------Load Dataset--------------------------------')
	margin = args.cubesize
	print('patch size ', margin)
	print('train stride ', args.stridet)
	split_comber = SplitComb(args.stridet, margin)

	dataset_train = data.AirwayData(
		config,
		phase='train',
		crop_size=args.cubesize,
		split_comber=split_comber,
		debug=args.debug,
		random_select=args.randsel)

	train_loader = DataLoader(
		dataset_train,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers,
		pin_memory=True)
	
	print('--------------------------------------')
	# split_comber = SplitComb(args.stridev, marginv)

	# # load validation dataset
	# dataset_val = data.AirwayData(
	# 	config,
	# 	phase='val',
	# 	split_comber=split_comber,
	# 	debug=args.debug,
	# 	random_select=False)
	# val_loader = DataLoader(
	# 	dataset_val,
	# 	batch_size=args.batch_size,
	# 	shuffle=False,
	# 	num_workers=args.workers,
	# 	pin_memory=True)

	print('--------------------------------------')

	##############################
	# start training
	##############################
	
	total_epoch = []
	train_loss,val_loss,test_loss = [],[],[]
	train_accrancy,val_accrancy,test_accrancy = [],[],[]
	train_sensitivity,val_sensitivity,test_sensitivity = [],[],[]
	train_DSC,val_DSC,test_DSC = [],[],[]
	train_precision,val_precision,test_precision = [],[],[]
	
	logdirpath = os.path.join(save_dir, 'log')
	if not os.path.exists(logdirpath):
		os.mkdir(logdirpath)

	v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = 0, 0, 0, 0, 0
	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = 0, 0, 0, 0, 0

	for epoch in range(start_epoch, end_epoch + 1):
		# 更新大小气道的采样频率
		t_loss, mean_accrancy, mean_sensitivity, mean_DSC, mean_precision = train_casenet(epoch, net, train_loader, optimizer, args, save_dir)
		train_loss.append(t_loss)
		train_accrancy.append(mean_accrancy)
		train_sensitivity.append(mean_sensitivity)
		train_DSC.append(mean_DSC)
		train_precision.append(mean_precision)

		# Save the current model
		if args.multigpu:
			state_dict = net.module.state_dict()
		else:
			state_dict = net.state_dict()
		for key in state_dict.keys():
			state_dict[key] = state_dict[key].cpu()
		torch.save({
			'state_dict': state_dict,
			'args': args},
			os.path.join(save_dir, 'latest.ckpt'))
		
		# Save the model frequently, default 5 epoch/save
		if epoch % args.save_freq == 0:            
			if args.multigpu:
				state_dict = net.module.state_dict()
			else:
				state_dict = net.state_dict()
			for key in state_dict.keys():
				state_dict[key] = state_dict[key].cpu()
			torch.save({
				'state_dict': state_dict,
				'args': args},
				os.path.join(save_dir, '%03d.ckpt' % epoch))

		# if (epoch % args.val_freq == 0) or (epoch == start_epoch):
		# 	v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = val_casenet(epoch, net, val_loader, args, save_dir)

		# if epoch % args.test_freq == 0:
		# 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = val_casenet(epoch, net, test_loader, args, save_dir, test_flag=True)
		
		val_loss.append(v_loss)
		val_accrancy.append(mean_acc2)
		val_sensitivity.append(mean_sensiti2)
		val_DSC.append(mean_dice2)
		val_precision.append(mean_ppv2)

		test_loss.append(te_loss)
		test_accrancy.append(mean_acc3)
		test_sensitivity.append(mean_sensiti3)
		test_DSC.append(mean_dice3)
		test_precision.append(mean_ppv3)
		
		total_epoch.append(epoch)

		totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss, train_accrancy, val_accrancy, test_accrancy,
							  train_sensitivity, val_sensitivity, test_sensitivity, train_DSC, val_DSC, test_DSC,
							  train_precision, val_precision, test_precision])
		np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)

	logName = os.path.join(logdirpath, 'log.csv')

	with open(logName, 'a') as csvout:
		writer = csv.writer(csvout)
		row =['train epoch', 'train loss', 'val loss', 'test loss', 'train accrancy', 'val accrancy', 'test accrancy',
	 		   'train sensitivity', 'val sensitivity', 'test sensitivity', 'DSC train', 'DSC val', 'DSC test',
	 		   'precision train','precision val', 'precision test']
		writer.writerow(row)

		for i in range(len(total_epoch)):
			row = [total_epoch[i], train_loss[i], val_loss[i], test_loss[i], 
				   train_accrancy[i], val_accrancy[i], test_accrancy[i],
				   train_sensitivity[i], val_sensitivity[i], test_sensitivity[i], 
				   train_DSC[i], val_DSC[i], test_DSC[i],
				   train_precision[i], val_precision[i], test_precision[i]]
			writer.writerow(row)
		csvout.close()

	print("Done")
	return


if __name__ == '__main__':
	main()

