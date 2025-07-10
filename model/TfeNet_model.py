# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
import DAConv_CUDA as da_cuda
import DSConv_CUDA as ds_cuda
import numpy as np

# Tubular Feature Fusion Module	
class TTFM(nn.Module):
	def __init__(self, in_ch, out_ch,kernel_size,padding,stride=1,scale=0.1,Conv="DACONV"):
		super(TTFM, self).__init__()
		self.conv_c = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
		if Conv == "DACONV":
			self.conv_x = da_cuda.DAConvPack(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,axis=0,scale=scale)
			self.conv_y = da_cuda.DAConvPack(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,axis=1,scale=scale)
			self.conv_z = da_cuda.DAConvPack(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,axis=2,scale=scale)
		elif Conv == "DSCONV":
			self.conv_x = ds_cuda.DSConvPack(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,axis=0,scale=scale)
			self.conv_y = ds_cuda.DSConvPack(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,axis=1,scale=scale)
			self.conv_z = ds_cuda.DSConvPack(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,axis=2,scale=scale)
		else:
			raise KeyError(f"Unsupported Conv type: {Conv}. Expected 'DACONV' or 'DSCONV'.")
		self.conv = nn.Conv3d(in_channels=4*out_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
		self.In_c = nn.InstanceNorm3d(out_ch)
		self.In_x = nn.InstanceNorm3d(out_ch)
		self.In_y = nn.InstanceNorm3d(out_ch)
		self.In_z = nn.InstanceNorm3d(out_ch)
		self.In_f = nn.InstanceNorm3d(out_ch)
		self.relu_c = nn.ReLU(inplace=True)
		self.relu_x = nn.ReLU(inplace=True)
		self.relu_y = nn.ReLU(inplace=True)
		self.relu_z = nn.ReLU(inplace=True)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
		self.In = nn.InstanceNorm3d(out_ch)


	def forward(self, x):
		input = x

		conv_c = self.conv_c(x)
		conv_c = self.In_c(conv_c)
		conv_c = self.relu_c(conv_c)

		conv_x = self.conv_x(x)
		conv_x = self.In_x(conv_x)
		conv_x = self.relu_x(conv_x)


		conv_y = self.conv_y(x)
		conv_y = self.In_y(conv_y)
		conv_y = self.relu_y(conv_y)


		conv_z = self.conv_z(x)
		conv_z = self.In_z(conv_z)
		conv_z = self.relu_z(conv_z)

		x = self.conv(torch.cat([conv_c,conv_x,conv_y,conv_z],dim=1))
		x = self.In_f(x)

		if x.shape != input.shape:
			input = self.conv1(input)
			input = self.In(input)

		return self.relu(x + input)

class ResConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(ResConv, self).__init__()
		self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
		self.In = nn.InstanceNorm3d(out_ch)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

	def forward(self, x):
		input = x
		
		x = self.conv(x)
		x = self.In(x)

		if x.shape != input.shape:
			input = self.conv1(input)
			input = self.In(input)

		return self.relu(x + input)
	
class TfeNet(nn.Module):
	def __init__(self, n_channels, number):
		super(TfeNet, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.number = number 

		self.conv1_1 = TTFM(n_channels,self.number,7,3,1,scale=0.1,Conv="DACONV")
		self.conv1_2 = TTFM(self.number,self.number,7,3,1,scale=0.1,Conv="DACONV")
		
		self.conv2_1 = TTFM(1*self.number,2*self.number,7,3,1,scale=0.1,Conv="DACONV")
		self.conv2_2 = TTFM(2*self.number,2*self.number,7,3,1,scale=0.1,Conv="DACONV")

		self.conv3_1 = TTFM(2*self.number,4*self.number,7,3,1,scale=0.1,Conv="DACONV")
		self.conv3_2 = TTFM(4*self.number,4*self.number,7,3,1,scale=0.1,Conv="DACONV")

		self.conv4_1 = TTFM(4*self.number,8*self.number,7,3,1,scale=0.1,Conv="DACONV")
		self.conv4_2 = TTFM(8*self.number,8*self.number,7,3,1,scale=0.1,Conv="DACONV")

		self.conv5_1 = TTFM(12*self.number,4*self.number,7,3,1,scale=1,Conv="DACONV")
		self.conv5_2 = TTFM(4*self.number,4*self.number,7,3,1,scale=1,Conv="DACONV")

		self.conv6_1 = TTFM(6*self.number,2*self.number,7,3,1,scale=1,Conv="DACONV")
		self.conv6_2 = TTFM(2*self.number,2*self.number,7,3,1,scale=1,Conv="DACONV")

		self.conv7_1 = ResConv(3*self.number,self.number)
		self.conv7_2 = ResConv(self.number,self.number)

		self.out_conv = nn.Conv3d(self.number, 1, 1)
		self.maxpooling = nn.MaxPool3d(2)
		self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		conv1_1 = self.conv1_1(x)
		conv1_2 = self.conv1_2(conv1_1)
		x = self.maxpooling(conv1_2)

		conv2_1 = self.conv2_1(x)
		conv2_2 = self.conv2_2(conv2_1)
		x = self.maxpooling(conv2_2)

		conv3_1 = self.conv3_1(x)
		conv3_2 = self.conv3_2(conv3_1)
		x = self.maxpooling(conv3_2)

		conv4_1 = self.conv4_1(x)
		conv4_2 = self.conv4_2(conv4_1)
		x = self.up(conv4_2)

		conv5_1 = self.conv5_1(torch.cat([x,conv3_2],dim=1))
		conv5_2 = self.conv5_2(conv5_1)
		x = self.up(conv5_2)

		conv6_1 = self.conv6_1(torch.cat([x,conv2_2],dim=1))
		conv6_2 = self.conv6_2(conv6_1)
		x = self.up(conv6_2)

		conv7_1 = self.conv7_1(torch.cat([x,conv1_2],dim=1))
		conv7_2 = self.conv7_2(conv7_1)

		x = self.out_conv(conv7_2)
		out = self.sigmoid(x)
		
		return out
