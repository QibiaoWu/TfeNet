import os
from time import time
import numpy as np
import data_CT_airways as data
from importlib import import_module
import shutil
from trainval_classifier_BAS import train_casenet, val_casenet, my_val_casenet
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
from local_logger import MyLocalLogger
from typing import Tuple, Union, List

warnings.filterwarnings("ignore")

def main(args):
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
			random_select=False,
			small_airway=args.small_airways)
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
			random_select=False,
			small_airway=args.small_airways)
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

	print("Loading train data...")

	dataset_train = data.AirwayData(
		config,
		phase='train',
		crop_size=args.cubesize,
		split_comber=split_comber,
		debug=args.debug,
		random_select=args.randsel,
		small_airway=args.small_airways)

	train_loader = DataLoader(
		dataset_train,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)
	
	print('--------------------------------------')
	split_comber_val = SplitComb(args.stridev, marginv)

	print("Loading val data...")
	# load validation dataset
	dataset_val = data.AirwayData(
		config,
		phase='val',
		split_comber=split_comber_val,
		debug=args.debug,
		random_select=False)
	val_loader = DataLoader(
		dataset_val,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)

	print('--------------------------------------')

	##############################
	# start training
	##############################
	
	print("Started training!")

	total_epoch = []
	train_loss,val_loss,test_loss = [],[],[]
	train_accuracy,val_accuracy,test_accuracy = [],[],[]
	train_sensitivity,val_sensitivity,test_sensitivity = [],[],[]
	train_DSC,val_DSC,test_DSC = [],[],[]
	train_precision,val_precision,test_precision = [],[],[]
	
	logdirpath = os.path.join(save_dir, 'log')
	if not os.path.exists(logdirpath):
		os.mkdir(logdirpath)

	v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = 0, 0, 0, 0, 0
	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = 0, 0, 0, 0, 0

	logName = os.path.join(logdirpath, 'log.csv')

	# check if log already exists
	if not os.path.exists(logName):
		# only write header if file is being created
		with open(logName, 'a') as csvout:
			writer = csv.writer(csvout)
			row = ['train epoch', 'train loss', 'val loss', 'test loss', 'train accuracy', 'val accuracy', 'test accuracy',
					'train sensitivity', 'val sensitivity', 'test sensitivity', 'DSC train', 'DSC val', 'DSC test',
					'precision train','precision val', 'precision test']
			writer.writerow(row)
			csvout.close()

	for epoch in range(start_epoch, end_epoch + 1):

		train_loader.dataset.shuffle_dataset()
		
		# 更新大小气道的采样频率
		t_loss, mean_accuracy, mean_sensitivity, mean_DSC, mean_precision = train_casenet(epoch, net, train_loader, optimizer, args, save_dir)
		train_loss.append(t_loss)
		train_accuracy.append(mean_accuracy)
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

		if (epoch % args.val_freq == 0) or (epoch == start_epoch):
			v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = val_casenet(epoch, net, val_loader, args, save_dir)

		# if epoch % args.test_freq == 0:
		# 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = val_casenet(epoch, net, test_loader, args, save_dir, test_flag=True)
		
		val_loss.append(v_loss)
		val_accuracy.append(mean_acc2)
		val_sensitivity.append(mean_sensiti2)
		val_DSC.append(mean_dice2)
		val_precision.append(mean_ppv2)

		test_loss.append(te_loss)
		test_accuracy.append(mean_acc3)
		test_sensitivity.append(mean_sensiti3)
		test_DSC.append(mean_dice3)
		test_precision.append(mean_ppv3)
		
		total_epoch.append(epoch)

		totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy,
							  train_sensitivity, val_sensitivity, test_sensitivity, train_DSC, val_DSC, test_DSC,
							  train_precision, val_precision, test_precision])
		np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)

		with open(logName, 'a') as csvout:
			writer = csv.writer(csvout)
			row = [epoch, t_loss, v_loss, te_loss, 
				mean_accuracy, mean_acc2, mean_acc3,
				mean_sensitivity, mean_sensiti2, mean_sensiti3, 
				mean_DSC, mean_dice2, mean_dice3,
				mean_precision, mean_ppv2, mean_ppv3]
			writer.writerow(row)
			csvout.close()

	print("Done")
	return


class EarlyStopping():

	def __init__(self, slope_window=15, early_stop_thresh=0.0005, early_stop_patience=30):
		self.slope_window = slope_window
		self.early_stop_thresh = early_stop_thresh
		self.early_stop_patience = early_stop_patience

	def check_slope(self, history, wait) -> Tuple[bool, float, int]:

		if len(history) < self.slope_window:
			return False, 1, 0 # Keep training

		# Get the last N points
		y = np.array(history[-self.slope_window:])
		# Create x-axis (0, 1, 2, 3, 4)
		x = np.arange(self.slope_window)
		
		# Perform linear regression (degree 1 polynomial)
		# polyfit returns [slope, intercept]
		slope, _ = np.polyfit(x, y, 1)
		
		# For Loss: If slope is flatter than positive threshold
		if slope is None:
			return False, 1, 0 # Keep training
		
		if slope < self.early_stop_thresh:
			wait += 1
		else:
			wait = 0 # Reset if we see a good upward trend

		if wait >= self.early_stop_patience:
			print(f"Early stopping triggered! Slope {slope:.6f} stayed below threshold for {self.early_stop_patience} epochs.")

			return True, slope, wait
		
		return False, slope, wait


def train_early_stop(args):
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
	print("Training with early stop!")

	# early stop parameteres
	prev_waiting_count = 0
	min_epoch = 60
	slope_window = 15

	iterations_val = 1000

	early_stopper = EarlyStopping(slope_window, 0.0005, 30)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	my_logger = MyLocalLogger(save_dir, args.resume)

	if not args.sgd:
		optimizer = optim.Adam(net.parameters(), lr=2e-2)  # args.lr
		# optimizer = optim.AdamW(net.parameters(), lr=0.01)
	else:
		optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

	if args.resume:
		resume_part = args.resumepart
		checkpoint = torch.load(args.resume)

		if resume_part:
			"""
			load part of the weight parameters
			"""
			net.load_state_dict(checkpoint['state_dict'], strict=False)
			my_logger.load_checkpoint(checkpoint['logging'])
			print('part load Done')
		else:
			"""
			load full weight parameters
			"""
			net.load_state_dict(checkpoint['state_dict'])
			my_logger.load_checkpoint(checkpoint['logging'])
			print("full resume Done")
		
		if 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("Optimizer state resumed")
		else:
			print("WARNING: No optimizer state found in checkpoint")
		
		start_epoch = len(my_logger.get_value('patience_waiting', step=None))
		# start_epoch = checkpoint.get('epoch', 0) + 1
		prev_waiting_count = my_logger.get_value('patience_waiting', step=-1)
	else:
		weights_init(net, init_type='xavier')  # weight initialization
	
	if args.epochs is None:
		end_epoch = args.lr_stage[-1]
	else:
		end_epoch = args.epochs
			
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

	target_device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	for state in optimizer.state.values():
		for k, v in state.items():
			if isinstance(v, torch.Tensor):
				state[k] = v.to(target_device)

	if args.test:
		print('---------------------testing---------------------')
		split_comber_tst = SplitComb(args.stridev, marginv)
		dataset_test = data.AirwayData(
			config,
			phase='test',
			split_comber=split_comber_tst,
			debug=args.debug,
			random_select=False,
			small_airway=args.small_airways)
		test_loader = DataLoader(
			dataset_test,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.workers,
			pin_memory=True)
		
		print('start testing')
		testdata = val_casenet(start_epoch, net, test_loader, args, save_dir, test_flag=True)
		return

	if args.debugval:
		print ('---------------------validation---------------------')
		split_comber = SplitComb(args.stridev, marginv)
		dataset_val = data.AirwayData(
			config,
			phase='val',
			split_comber=split_comber,
			debug=args.debug,
			random_select=False,
			small_airway=args.small_airways)
		val_loader = DataLoader(
			dataset_val,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.workers,
			pin_memory=True)
		valdata = val_casenet(start_epoch, net, val_loader, args, save_dir)
		return

	print('---------------------------------Load Dataset--------------------------------')
	margin = args.cubesize
	print('patch size ', margin)
	print('train stride ', args.stridet)
	split_comber = SplitComb(args.stridet, margin)

	print("Loading train data...")

	dataset_train = data.AirwayData(
		config,
		phase='train',
		crop_size=args.cubesize,
		split_comber=split_comber,
		debug=args.debug,
		random_select=args.randsel,
		small_airway=args.small_airways)

	train_loader = DataLoader(
		dataset_train,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)
	
	print('-------------------Load Validation-------------------')
	split_comber_val = SplitComb(args.stridev, marginv)
	print('patch size ', marginv)
	print('val stride ', args.stridev)
	print("Loading val data...")

	# load validation dataset
	dataset_val = data.AirwayData(
		config,
		phase='val',
		split_comber=split_comber_val,
		debug=args.debug,
		random_select=args.randsel)
	val_loader = DataLoader(
		dataset_val,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)

	print('--------------------------------------')

	##############################
	# start training
	##############################

	v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = 0, 0, 0, 0, 0
	# te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = 0, 0, 0, 0, 0

	print("Started training...")
	
	stop = False
	for epoch in range(start_epoch, end_epoch):
		
		train_loader.dataset.shuffle_dataset()
		
		my_logger.log('epoch_start_timestamps', time(), epoch)
		t_loss, mean_accuracy, mean_sensitivity, mean_DSC, mean_precision, lrs = train_casenet(epoch, net, train_loader, optimizer, args, save_dir)
		my_logger.log('epoch_end_timestamps', time(), epoch)

		# if (epoch % args.val_freq == 0) or (epoch == start_epoch):
		v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = my_val_casenet(epoch, net, val_loader, args, iterations_val)

		# if epoch % args.test_freq == 0:
		# 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = val_casenet(epoch, net, test_loader, args, save_dir, test_flag=True)

		# test_loss.append(te_loss)
		# test_accuracy.append(mean_acc3)
		# test_sensitivity.append(mean_sensiti3)
		# test_DSC.append(mean_dice3)
		# test_precision.append(mean_ppv3)
		# 
		# total_epoch.append(epoch)

		my_logger.log('train_loss', t_loss, epoch)
		my_logger.log('train_acc', mean_accuracy, epoch)
		my_logger.log('train_sensitivity', mean_sensitivity, epoch)
		my_logger.log('train_dice', mean_DSC, epoch)
		my_logger.log('train_precision', mean_precision, epoch)
		my_logger.log('val_loss', v_loss, epoch)
		my_logger.log('val_acc', mean_acc2, epoch)
		my_logger.log('val_sensitivity', mean_sensiti2, epoch)
		my_logger.log('val_dice', mean_dice2, epoch)
		my_logger.log('val_precision', mean_ppv2, epoch)
		my_logger.log('lrs', lrs, epoch)

		# treat early stop
		if epoch > min_epoch:
			stop, slope, prev_waiting_count = early_stopper.check_slope(my_logger.get_last_x_values('ema_val_dice', last=slope_window), prev_waiting_count) # if early stopping
			my_logger.log('patience_waiting', prev_waiting_count, epoch)
			print(f"Debug: Slope {slope:.6f} \t Patience count {prev_waiting_count}")
            
		else:
			my_logger.log('patience_waiting', 0, epoch)
		
		my_logger.log('patience_waiting', prev_waiting_count, epoch)
		
		# Save the current model, default every 5 epoch/save
		if epoch % args.save_freq == 0:
			if args.multigpu:
				state_dict = net.module.state_dict()
			else:
				state_dict = net.state_dict()
			
			for key in state_dict.keys():
				state_dict[key] = state_dict[key].cpu()
			
			torch.save({
				'state_dict': state_dict,
				'optimizer': optimizer.state_dict(),
				'args': args,
				'epoch': epoch,
				'logging': my_logger.get_checkpoint()},
				os.path.join(save_dir, 'latest.ckpt'))
			
			my_logger.plot_progress_png(save_dir)
		
		# save final model
		if epoch == (end_epoch - 1) or stop:
			if args.multigpu:
				state_dict = net.module.state_dict()
			else:
				state_dict = net.state_dict()
			
			for key in state_dict.keys():
				state_dict[key] = state_dict[key].cpu()
			
			torch.save({
				'state_dict': state_dict,
				'optimizer': optimizer.state_dict(),
				'args': args,
				'epoch': epoch,
				'logging': my_logger.get_checkpoint()},
				os.path.join(save_dir, f"final_{end_epoch}.ckpt"))

		if stop:
			break

	print("Done")
	return


if __name__ == '__main__':
	global args
	args = parser.parse_args()

	if args.early_stop:
		train_early_stop(args)
	else:
		main(args)

