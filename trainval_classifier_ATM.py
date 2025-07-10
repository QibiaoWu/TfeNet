import numpy as np
import os
import time
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils import save_itk, load_itk_image, DSC_np, precision_np,\
sensitivity_np, accrancy_np, combine_total_avg, combine_total, normalize_min_max
from loss import TverskyLoss,BoundaryDoULoss3D,tversky_loss_2,dice_loss, binary_cross_entropy, focal_loss, general_union_loss,tversky_loss
from torch.cuda import empty_cache
import csv
from scipy.ndimage.interpolation import zoom
import gc
import torch.nn as nn
from monai.transforms import distance_transform_edt


th_bin = 0.5
epsilon = 1e-6


def get_lr(epoch, args):
	"""
	:param epoch: current epoch number
	:param args: global arguments args
	:return: learning rate of the next epoch
	"""
	if args.lr is None:
		assert epoch <= args.lr_stage[-1]
		lrstage = np.sum(epoch > args.lr_stage)
		lr = args.lr_preset[lrstage]
	else:
		lr = args.lr
	return lr


def train_casenet(epoch, model, data_loader, optimizer, args, save_dir):
	"""
	:param epoch: current epoch number
	:param model: CNN model
	:param data_loader: training data
	:param optimizer: training optimizer
	:param args: global arguments args
	:param save_dir: save directory
	:return: performance evaluation of the current epoch
	"""
	model.train()
	starttime = time.time()
	sidelen = args.stridet
	margin = args.cubesize
	if args.device == 1:
		device = ['cuda:1']
	else: 
		device = ['cuda:0']

	lr = get_lr(epoch, args)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	assert (lr is not None)
	optimizer.zero_grad()

	loss_total = []
	DSC_total = []
	precision_total = []
	accrancy_total = []
	DSC_hard_total = []
	sensitivity_total = []
	traindir = os.path.join(save_dir, 'train')
	if not os.path.exists(traindir):
		os.mkdir(traindir)
	
	for i, (x, y, weight) in enumerate(tqdm(data_loader)):
		######Wrap Tensor##########
		batchlen = x.size(0)
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
			weight = weight.cuda()
			weight = weight.cuda()
		###############################

		casePred = model(x) # baseline
		
		r = (np.random.randint(0,5)/5 + 2)
		weight = 0.95*(weight**r) + 0.05

		loss = general_union_loss(casePred, y, weight, alpha=0.05) #+ lsd_Loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# for evaluation
		loss_total.append(loss.item())
		# segmentation calculating metrics#######################
		with torch.no_grad():
			outdata = casePred.cpu().data.numpy()
			segdata = y.cpu().data.numpy()

			segdata = (segdata > th_bin)
			segpred = (outdata > th_bin)

			for j in range(batchlen):
				dice = DSC_np(outdata[j, 0], segdata[j, 0])
				dicehard = DSC_np(segpred[j, 0], segdata[j, 0])
				ppv = precision_np(segpred[j, 0], segdata[j, 0])
				sensiti = sensitivity_np(segpred[j, 0], segdata[j, 0])
				acc = accrancy_np(segpred[j, 0], segdata[j, 0])

				##########################################################################
				DSC_total.append(dice)
				precision_total.append(ppv)
				sensitivity_total.append(sensiti)
				accrancy_total.append(acc)
				DSC_hard_total.append(dicehard)

	##################################################################################
	
	endtime = time.time()
	loss_total = np.array(loss_total)
	mean_DSC = np.mean(np.array(DSC_total))
	mean_DSC_hard = np.mean(np.array(DSC_hard_total))
	mean_precision = np.mean(np.array(precision_total))
	mean_sensitivity = np.mean(np.array(sensitivity_total))
	mean_accrancy = np.mean(np.array(accrancy_total))
	mean_loss = np.mean(loss_total)

	print('Train, epoch %d, loss %.4f, accuracy %.4f, sensitivity %.4f, DSC %.4f, DSC hard %.4f, precision %.4f, time %3.2f, lr % .5f '
		  %(epoch, mean_loss, mean_accrancy, mean_sensitivity, mean_DSC, mean_DSC_hard, mean_precision, endtime-starttime,lr))
	print ()
	empty_cache()
	return mean_loss, mean_accrancy, mean_sensitivity, mean_DSC, mean_precision

def val_casenet(epoch, model, data_loader, args, save_dir, test_flag=False):
	"""
	:param epoch: current epoch number
	:param model: CNN model
	:param data_loader: evaluation and testing data
	:param args: global arguments args
	:param save_dir: save directory
	:param test_flag: current mode of validation or testing
	:return: performance evaluation of the current epoch
	"""
	model.eval()

	sidelen = args.stridev
	if args.cubesizev is not None:
		margin = args.cubesizev
	else:
		margin = args.cubesize


	if test_flag:
		valdir = r'/workspace/outputs'
		state_str = 'test'

	if not os.path.exists(valdir):
		os.mkdir(valdir)

	p_total = {}

	with torch.no_grad():
		for i, (x, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(tqdm(data_loader)):
			######Wrap Tensor##########
			NameID = NameID[0]
			SplitID = SplitID[0] 
			batchlen = x.size(0)
			if torch.cuda.is_available():
				x = x.cuda()
			####################################################

			casePred = model(x) # baseline-DSCNet
		
			#####################seg data#######################
			outdata = casePred.cpu().data.numpy()
			#######################################################################
			origindata = org.numpy()
			spacingdata = spac.numpy()
			#######################################################################
			#################REARRANGE THE DATA BY SPLIT ID########################
			for j in range(batchlen):
				# curxdata = (xdata[j, 0]*255)
				segpred = outdata[j, 0]
				curorigin = origindata[j].tolist()
				curspacing = spacingdata[j].tolist()
				cursplitID = int(SplitID[j])
				assert (cursplitID >= 0)
				curName = NameID[j]
				curnzhw = nzhw[j]
				curshape = ShapeOrg[j]

				# 添加新的字典对象，以验证样本的名字进行索引
				if not (curName in p_total.keys()):
					p_total[curName] = []

				curpinfo = [segpred, cursplitID, curnzhw, curshape, curorigin, curspacing]
			
				p_total[curName].append(curpinfo)
				del curpinfo,segpred,curorigin,curspacing,cursplitID,curName,curnzhw,curshape
			del outdata, origindata, spacingdata  # Clear unnecessary data

	# combine all the cases together
	for curName in p_total.keys():
		curp = p_total[curName]
		p_combine, porigin, pspacing = combine_total_avg(curp, sidelen, margin)
		p_combine_bw = (p_combine > th_bin)
		curpredpath = os.path.join(valdir, '%s.nii.gz'%(curName))
		save_itk(p_combine_bw.astype(dtype='uint8'), porigin, pspacing, curpredpath)

		del  curp, p_combine_bw, p_combine
		gc.collect()




