#!/usr/bin/env python  
# encoding: utf-8  
import argparse

parser = argparse.ArgumentParser(description='PyTorch Airway Segmentation')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline', help='model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
					metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--save-freq', default='5', type=int, metavar='S',
					help='save frequency')
parser.add_argument('--val-freq', default='10', type=int, metavar='S',
					help='validation frequency')
parser.add_argument('--test-freq', default='10', type=int, metavar='S',
					help='testing frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--resumepart', default=0, type=int, metavar='PARTRESUME',
					help='Resume params. part')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
					help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
					help='1 do test evaluation, 0 not')
parser.add_argument('--debug', default=0, type=int, metavar='TEST',
					help='debug mode')
parser.add_argument('--randsel', default=0, type=int, metavar='RandomSelect',
					help='randomly select samples for training')
parser.add_argument('--debugval', default=0, type=int, metavar='Validation',
					help='debug mode for validation')
parser.add_argument('--sgd', default=0, type=int, metavar='SGDopti',
					help='use sgd')
parser.add_argument('--cubesize', default=[80, 192, 304], nargs="*", type=int, metavar='cube',
					help='cube size')
parser.add_argument('--cubesizev', default=None,nargs="*", type=int, metavar='cube',
					help='cube size')
parser.add_argument('--stridet', default=[48, 80, 80], nargs="*", type=int, metavar='stride',
					help='split stride train')
parser.add_argument('--stridev', default=[48, 80, 80], nargs="*", type=int, metavar='stride',
					help='split stride val')
parser.add_argument('--multigpu', default=False, type=bool, metavar='mgpu',
					help='use multiple gpus')
parser.add_argument('--device', default=0, type=int, metavar='device',
					help='cuda device number')


