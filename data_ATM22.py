import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
import SimpleITK as sitk
from utils import load_itk_image
import scipy.ndimage as ndimage

np.random.seed(777) #numpy

class AirwayData(Dataset):
	"""
	Generate dataloader
	"""
	def __init__(self, config, phase='train', stage=1, split_comber=None,
				 debug=False, crop_size=[128, 128, 128],random_select=False):
		"""
		:param config: configuration from model
		:param phase: training or validation or testing
		:param split_comber: split-combination-er
		:param debug: debug mode to check few data
		:param random_select: use partly, randomly chosen data for training
		"""
		assert(phase == 'train' or phase == 'val' or phase == 'test')
		self.phase = phase
		self.augtype = config['augtype']
		self.split_comber = split_comber
		self.patch_per_case = 16  # patches used per case 
		self.debug_flag = debug
		self.stage = stage
		self.crop_size = crop_size


		"""
		specify the path and data split
		"""
		self.datapath = config['dataset_path']

		print("-------------------------Load all data into memory---------------------------")
		"""
		count the number of cases
		"""

		cubelist = []
		self.caseNumber = 0
		self.label_list = []
		self.weight_list = []
		self.img_list = []
		self.allimgdata_memory = {}

		if self.phase == 'train':
			# train_path = os.path.join(self.datapath + '\\image\\train')	
			train_path = os.path.join(self.datapath + '/image_clean/train')
			file_name_list = os.listdir(train_path)
			file_num = len(file_name_list)
			if self.debug_flag:
				file_name_list = file_name_list[:1]
				file_num = len(file_name_list)
			self.caseNumber += file_num

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in file_name_list:
				raw_path = os.path.join(train_path, file_name)
				# data_name = raw_path.split('\\')[-1].split('.nii')[0]
				data_name = raw_path.split('/')[-1].split('.nii')[0]

				label_path=raw_path.replace('image', 'label')
				label_path=label_path.replace('clean_hu', 'label')

				weight_path=raw_path.replace('image_clean', 'LIB_weight')
				weight_path=weight_path.replace('clean_hu.nii.gz', 'weight.npy')


				self.img_list.append(raw_path)
				self.label_list.append(label_path)
				self.weight_list.append(weight_path)

		elif self.phase == 'val':
		
			val_path = os.path.join(self.datapath + '/image_clean/val01')
			self.img_list = os.listdir(val_path)
			file_num = len(self.img_list)
			if self.debug_flag:
				self.img_list = self.img_list[:1]
				file_num = len(self.img_list)
			self.caseNumber += file_num

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in self.img_list:
				raw_path = os.path.join(val_path, file_name)
				imgs, origin, spacing = load_itk_image(raw_path)
				splits, nzhw, orgshape = self.split_comber.split_id(imgs)
				data_name = raw_path.split('/')[-1].split('.nii')[0]
				print("Name: %s, # of splits: %d"%(data_name, len(splits)))
				
				for j in range(len(splits)):
					cursplit = splits[j]
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
					cubelist.append(curlist)

				self.allimgdata_memory[data_name] = [imgs, origin, spacing]
		elif self.phase == 'test':
		
			test_path = os.path.join(self.datapath)
			self.img_list = os.listdir(test_path)
			file_num = len(self.img_list)
			if self.debug_flag:
				self.img_list = self.img_list[:1]
				file_num = len(self.img_list)
			self.caseNumber += file_num

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in self.img_list:
				raw_path = os.path.join(test_path, file_name)
				imgs, origin, spacing = load_itk_image(raw_path)
				splits, nzhw, orgshape = self.split_comber.split_id(imgs)
				data_name = raw_path.split('/')[-1].split('.nii')[0]
				print("Name: %s, # of splits: %d"%(data_name, len(splits)))
				
				for j in range(len(splits)):
					cursplit = splits[j]
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
					cubelist.append(curlist)
				self.allimgdata_memory[data_name] = raw_path

		if self.phase == 'val':
			random.shuffle(cubelist)
			self.cubelist = cubelist
		elif self.phase == 'test':
			self.cubelist = cubelist

		print('---------------------Initialization Done---------------------')
		# print('Phase: %s total cubelist number: %d'%(self.phase, len(self.cubelist)))
		print()

	def __len__(self):
		"""
		:return: length of the dataset
		"""
		if self.phase == 'train':
			return self.patch_per_case*self.caseNumber
		else:
			return len(self.cubelist)

	def __getitem__(self, idx):
		"""
		:param idx: index of the batch
		:return: wrapped data tensor and name, shape, origin, etc.
		"""
		t = time.time()
		np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
		if self.phase != 'train':
			curlist = self.cubelist[idx]
			curNameID = curlist[0]
			cursplit = curlist[1]
			curSplitID = curlist[2]
			curnzhw = curlist[3]
			curShapeOrg = curlist[4]
		else:
			list_index = idx//self.patch_per_case

		if self.phase != 'train':
			# imginfo = self.allimgdata_memory[curNameID]
			# imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
			imgs, origin, spacing = load_itk_image(self.allimgdata_memory[curNameID])
			imgs = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
			# 处理HU
			imgs = lumTrans_hu(imgs)
			imgs = (imgs.astype(np.float32))/255.0 # 数据增强后进行归一化
		else :
			img_path = self.img_list[list_index]
			label_path = self.label_list[list_index]
			weight_path = self.weight_list[list_index]


			imgs, origin, spacing = load_itk_image(img_path)
			label,_,_ = load_itk_image(label_path)
			label = (label > 0)
			label = label.astype('float')
			weight = np.load(weight_path)

			if imgs.shape[0] <= 145 or imgs.shape[1] <= 145 or imgs.shape[2] <= 145:
				imgs, label, weight = random_sample(imgs, label, weight, [128,128,128])
			else:
				imgs, label, weight = random_sample(imgs, label, weight, [145,145,145])
				if  self.augtype['rotate'] is True:
					imgs, label, weight = augment_random_rotate(imgs, label, weight, angle=10,threshold=0.7)
					imgs, label, weight = central_crop(imgs, label, weight, self.crop_size)

			imgs = (imgs.astype(np.float32))/255.0 # 数据增强后进行归一化
			
		####################################################################
		imgs = imgs[np.newaxis,...]
		#######################################################################

		if self.phase == 'train':
			weight = weight[np.newaxis,...]
			label = label[np.newaxis,...]
			return torch.from_numpy(imgs).float(),torch.from_numpy(label).float(),\
			torch.from_numpy(weight).float()
		else:
			curNameID = [curNameID]
			curSplitID = [curSplitID]
			curnzhw = np.array(curnzhw)
			curShapeOrg = np.array(curShapeOrg)
			return torch.from_numpy(imgs).float(),\
				torch.from_numpy(origin),\
				torch.from_numpy(spacing), curNameID, curSplitID,\
				torch.from_numpy(curnzhw),torch.from_numpy(curShapeOrg)


class SegValData(Dataset):
	def __init__(self, file_path):
		list = os.listdir(file_path)
		self.file_list = []
		for file in list:
			path = os.path.join(file_path,file)
			self.file_list.append(path)

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, item):
		name = self.file_list[item].split('/')[-1]
		img,origin,spacing = load_itk_image(self.file_list[item])

		img = lumTrans_hu(img)
		img = (img.astype(np.float32))/255.0
		# name = [name]
		img = img[np.newaxis, ...]
		return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(origin),\
				torch.from_numpy(spacing),name
			


def augment_random_rotate(img, label, weight, angle,threshold):
	rotate_angle = np.random.randint(angle)*np.sign(np.random.random()-0.5)
	rotate_axes = [(0,1),(1,2),(0,2)]
	k = np.random.randint(0,3)
	img = ndimage.rotate(img, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
	label = label.astype(np.float32)
	label = ndimage.rotate(label, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
	threshold = threshold   #threshold=0.7 in stage1 and 0.9 in stage2
	label[label>=threshold] = 1 
	label[label<threshold] = 0
	label = label.astype(np.uint8)
	weight = weight.astype(np.float32)
	weight = ndimage.interpolation.rotate(weight, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
	weight[weight>1] = 1
	weight[weight<0] = 0

	img[img<0] = 0.
	img[img>255] = 255.0
	img = img.astype(np.uint8)
		
	return img, label, weight

def central_crop(sample, label, weight, crop_size):
	origin_size = sample.shape
	crop_size = np.array(crop_size)
	start = (origin_size - crop_size)//2
	sample = sample[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
	label = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
	weight = weight[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]

	return sample, label, weight

def random_sample(img, label, weight, crop_size):
	origin_size = img.shape

	start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
			np.random.randint(0, origin_size[2] - crop_size[2])]

	img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
				start[2]:(start[2] + crop_size[2])]
	label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
					start[2]:start[2] + crop_size[2]]
	weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
				start[2]:start[2] + crop_size[2]]

	
	return img_crop, label_crop, weight_crop

def lumTrans_hu(img):
	"""
	:param img: CT image
	:return: Hounsfield Unit window clipped and normalized
	"""
	img[np.isnan(img)] = -2000
	lungwin = np.array([-1000.,600.])
	newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	newimg = (newimg*255).astype('uint8')
	return newimg

