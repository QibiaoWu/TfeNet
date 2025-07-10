#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


epsilon = 1e-6


def dice_loss(pred, target):
	"""
	DSC loss
	: param pred: input prediction
	: param target: input target
	"""
	smooth = 1e-6
	iflat = pred.reshape(-1)
	tflat = target.reshape(-1)
	intersection = torch.sum((iflat * tflat))
	return 1. - ((2. * intersection + smooth)/(torch.sum(iflat) + torch.sum(tflat) + smooth))


def binary_cross_entropy(y_pred, y_true,weight):
	"""
	Binary cross entropy loss
	: param y_pred: input prediction
	: param y_true: input target
	"""
	weight = y_true*weight + (1-y_true)
	weight = weight.view(-1).float()
	y_true = y_true.view(-1).float()
	y_pred = y_pred.view(-1).float()
	return F.binary_cross_entropy(y_pred, y_true,weight)


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
	"""
	Focal loss
	: param y_pred: input prediction
	: param y_true: input target
	: param alpha: balancing positive and negative samples, default=0.25
	: param gamma: penalizing wrong predictions, default=2
	"""
	# alpha balance weight for unbalanced positive and negative samples
	# clip to prevent NaN's and Inf's
	y_pred_flatten = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
	y_pred_flatten = y_pred_flatten.view(-1).float()
	y_true_flatten = y_true.detach()
	y_true_flatten = y_true_flatten.view(-1).float()
	loss = 0

	idcs = (y_true_flatten > 0)
	y_true_pos = y_true_flatten[idcs]
	y_pred_pos = y_pred_flatten[idcs]
	y_true_neg = y_true_flatten[~idcs]
	y_pred_neg = y_pred_flatten[~idcs]

	if y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0:
		# positive samples
		logpt = torch.log(y_pred_pos)
		loss += -1. * torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha

	if y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0:
		# negative samples
		logpt2 = torch.log(1. - y_pred_neg)
		loss += -1. * torch.mean(torch.pow(y_pred_neg, gamma) * logpt2) * (1. - alpha)

	return loss

def general_union_loss(pred, target, weight, alpha=0.1):
	if weight is None:
		weight = torch.ones_like(target)
	else:
		weight = weight*target + (1-target)
	#when weight = 1, this loss becomes Root Tversky loss
	smooth = 0	
	# alpha = 0.1 #alpha=0.1 in stage1 and 0.2 in stage2
	beta = 1 - alpha
	sigma1 = 1e-8
	sigma2 = 1e-8
	weight_i = target*sigma1 + (1-target)*sigma2
	intersection = (weight*((pred+weight_i)**0.7)*target).sum()
	intersection2 = (weight*(alpha*pred + beta*target)).sum()
	return 1-(intersection + smooth)/(intersection2 + smooth)

def tversky_loss(pred, target,alpha=0.1):
	smooth = 1e-6
	# alpha = 0.2
	beta = 1 - alpha
	intersection = (pred * target).sum()
	FP = (pred * (1 - target)).sum()
	FN = ((1 - pred) * target).sum()
	return 1 - (intersection+smooth) / (intersection + alpha * FP + beta * FN + smooth)

def BoundaryDoULoss3D(inputs, target,alpha):
	smooth = 1e-4
	alpha = torch.clamp(alpha, 0,0.8)  # Truncate alpha value
	loss = torch.sum(target+inputs**2 - 2*target*inputs + smooth)/torch.sum(target+inputs**2 - (1+alpha)*target*inputs + smooth)
	return loss
