import torch
import torch.nn as nn
import numpy as np
from model.TfeNet_model import TfeNet


config = {'pad_value': 0,     
		  'augtype': {'rotate': True},
		  'startepoch': 0, 'lr_stage': np.array([20, 40, 60, 70]), 'lr': np.array([1e-2, 1e-3, 1e-4,1e-5]),	
          'dataset_path': r'/home/wqb/wqb/dataset/BAS'}


def get_model(args=None):

	net = TfeNet(n_channels=1,number=16) 

	print('# of network parameters:', sum(param.numel() for param in net.parameters()))
	return config, net


if __name__ == '__main__':
	_, model = get_model()
