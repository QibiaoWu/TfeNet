import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
import SimpleITK as sitk
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from utils import load_itk_image
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize_3d
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
		labellist = []
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}
		alllabeldata_memory = {}
		allweightdata_memory = {}
		allsmallairwaydata_memory = {}

		if self.phase == 'train':
			# train_path = os.path.join(self.datapath + '\\image\\train')	
			train_path = os.path.join(self.datapath + '/image_clean/train')
			data_file_names = os.listdir(train_path)
			file_num = len(data_file_names)
			if self.debug_flag:
				data_file_names = data_file_names[:1]
				file_num = len(data_file_names)
			self.caseNumber += file_num
			self.data_file_names = data_file_names

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in data_file_names:
				raw_path = os.path.join(train_path, file_name)
				# data_name = raw_path.split('\\')[-1].split('.nii')[0]
				data_name = raw_path.split('/')[-1].split('.nii')[0]

				label_path=raw_path.replace('image', 'label')
				label_path=label_path.replace('clean_hu', 'label')

				imgs, origin, spacing = load_itk_image(raw_path)
				labels, _, _ = load_itk_image(label_path)

				# small_path=raw_path.replace('image_clean', 'smallairway_clean')
				# small_path=small_path.replace('clean_hu', 'smallairway')
				# small_labels, _, _  = load_itk_image(small_path)

				weight_path=raw_path.replace('image_clean', 'LIB_weight')
				weight_path=weight_path.replace('clean_hu.nii.gz', 'weight.npy')
				weight = np.load(weight_path)


				allimgdata_memory[data_name] = [imgs, origin, spacing]
				alllabeldata_memory[data_name] = labels
				allweightdata_memory[data_name] = weight
				# allsmallairwaydata_memory[data_name] = small_labels
				print("Name: %s "%(data_name),imgs.shape)
		

		elif self.phase == 'val':
			# val_path = os.path.join(self.datapath + '\\image\\val')
			val_path = os.path.join(self.datapath + '/image_clean/val')
			data_file_names = os.listdir(val_path)
			file_num = len(data_file_names)
			if self.debug_flag:
				data_file_names = data_file_names[:1]
				file_num = len(data_file_names)
			self.caseNumber += file_num

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in data_file_names:
				raw_path = os.path.join(val_path, file_name)
				label_path=raw_path.replace('image', 'label')
				label_path=label_path.replace('clean_hu', 'label')
				imgs, origin, spacing = load_itk_image(raw_path)
				splits, nzhw, orgshape = self.split_comber.split_id(imgs)
				data_name = raw_path.split('/')[-1].split('.nii')[0]
				# data_name = raw_path.split('\\')[-1].split('.nii')[0]
				print("Name: %s, # of splits: %d"%(data_name, len(splits)))
				labels, _, _ = load_itk_image(label_path)
				# small_path=raw_path.replace('image_clean', 'smallairway_clean')
				# small_path=small_path.replace('clean_hu', 'smallairway')
				# small_labels, _, _  = load_itk_image(small_path)
				
				allimgdata_memory[data_name] = [imgs, origin, spacing]
				alllabeldata_memory[data_name] = labels
				# allsmallairwaydata_memory[data_name] = small_labels

				for j in range(len(splits)):
					cursplit = splits[j]
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
					cubelist.append(curlist)

		else:
			# test_path = os.path.join(self.datapath + '\\image\\test')
			test_path = os.path.join(self.datapath + '/image_clean/test')
			data_file_names = os.listdir(test_path)
			file_num = len(data_file_names)
			if self.debug_flag:
				data_file_names = data_file_names[:1]
				file_num = len(data_file_names)
			self.caseNumber += file_num
			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in data_file_names:
				raw_path = os.path.join(test_path, file_name)
				label_path=raw_path.replace('image', 'label')
				label_path=label_path.replace('clean_hu', 'label')
				imgs, origin, spacing = load_itk_image(raw_path)
				# imgs = lumTrans(imgs) # 设置窗宽窗位为[-1000,600]
				splits, nzhw, orgshape = self.split_comber.split_id(imgs)
				data_name = raw_path.split('/')[-1].split('.nii')[0]
				# data_name = raw_path.split('\\')[-1].split('.nii')[0]
				print("Name: %s, # of splits: %d"%(data_name, len(splits)))
				labels, _, _ = load_itk_image(label_path)

				# small_path=raw_path.replace('image_clean', 'smallairway_clean')
				# small_path=small_path.replace('clean_hu', 'smallairway')
				# small_labels, _, _  = load_itk_image(small_path)

				allimgdata_memory[data_name] = [imgs, origin, spacing]
				alllabeldata_memory[data_name] = labels
				# allsmallairwaydata_memory[data_name] = small_labels
				

				for j in range(len(splits)):
					"""
					check if this cube is suitable
					"""
					cursplit = splits[j]
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
					cubelist.append(curlist)

		self.allimgdata_memory = allimgdata_memory
		self.alllabeldata_memory = alllabeldata_memory
		self.allweightdata_memory = allweightdata_memory
		# self.allsmallairwaydata_memory = allsmallairwaydata_memory
		del allimgdata_memory,alllabeldata_memory,allweightdata_memory,allsmallairwaydata_memory
		if self.phase == 'train':
			random.shuffle(data_file_names)
		elif self.phase == 'val':
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
			curtransFlag = curlist[5]
		else:
			data_name=self.data_file_names[idx//self.patch_per_case]
			curNameID = data_name.split('.nii')[0]
			curtransFlag = 'Y'

		label = self.alllabeldata_memory[curNameID]
		label = (label > 0)
		label = label.astype('float')

		# small_labels = self.allsmallairwaydata_memory[curNameID]
		# small_labels = (small_labels > 0)
		# small_labels = small_labels.astype('float')
			
		if self.phase != 'train':
			label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]	
			# small_labels = small_labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]	
		####################################################################
		
		if self.phase != 'train':
			imginfo = self.allimgdata_memory[curNameID]
			imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
			imgs = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
			imgs = (imgs.astype(np.float32))/255.0 
		else :
			weight = self.allweightdata_memory[curNameID]
			imginfo = self.allimgdata_memory[curNameID]
			imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]

			
			imgs, label, weight = random_sample(imgs, label, weight, [145,145,145])
			
			if  curtransFlag == 'Y' and self.augtype['rotate'] is True:
				imgs, label, weight = augment_random_rotate(imgs, label, weight, angle=10,threshold=0.7)
				imgs, label, weight = central_crop(imgs, label, weight, self.crop_size)
			
			imgs = (imgs.astype(np.float32))/255.0 # 数据增强后进行归一化
		
		####################################################################

		imgs = imgs[np.newaxis,...]
		label = label[np.newaxis,...]
		
		if self.phase == 'train':
			weight = weight[np.newaxis,...]
			return torch.from_numpy(imgs).float(),torch.from_numpy(label).float(),\
				torch.from_numpy(weight).float(),curNameID
		else:
			curNameID = [curNameID]
			curSplitID = [curSplitID]
			curnzhw = np.array(curnzhw)
			curShapeOrg = np.array(curShapeOrg)
			return torch.from_numpy(imgs).float(),torch.from_numpy(label).float(),\
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
			


def augment_split_jittering(cursplit, curShapeOrg):
	# orgshape [z, h, w]
	zstart, zend = cursplit[0][0], cursplit[0][1]
	hstart, hend = cursplit[1][0], cursplit[1][1]
	wstart, wend = cursplit[2][0], cursplit[2][1]
	curzjitter, curhjitter, curwjitter = 0, 0, 0
	if zend - zstart <= 3:
		jitter_range = (zend - zstart) * 32
	else:
		jitter_range = (zend - zstart) * 2
	# print("jittering range ", jitter_range)
	jitter_range_half = jitter_range//2

	t = 0
	while t < 10:
		if zstart == 0:
			curzjitter = int(np.random.rand() * jitter_range)
		elif zend == curShapeOrg[0]:
			curzjitter = -int(np.random.rand() * jitter_range)
		else:
			curzjitter = int(np.random.rand() * jitter_range) - jitter_range_half
		t += 1
		if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
			break

	t = 0
	while t < 10:
		if hstart == 0:
			curhjitter = int(np.random.rand() * jitter_range)
		elif hend == curShapeOrg[1]:
			curhjitter = -int(np.random.rand() * jitter_range)
		else:
			curhjitter = int(np.random.rand() * jitter_range) - jitter_range_half
		t += 1
		if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
			break

	t = 0
	while t < 10:
		if wstart == 0:
			curwjitter = int(np.random.rand() * jitter_range)
		elif wend == curShapeOrg[2]:
			curwjitter = -int(np.random.rand() * jitter_range)
		else:
			curwjitter = int(np.random.rand() * jitter_range) - jitter_range_half
		t += 1
		if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
			break

	if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
		cursplit[0][0] = curzjitter + zstart
		cursplit[0][1] = curzjitter + zend

	if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
		cursplit[1][0] = curhjitter + hstart
		cursplit[1][1] = curhjitter + hend

	if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
		cursplit[2][0] = curwjitter + wstart
		cursplit[2][1] = curwjitter + wend
	# print ("after ", cursplit)
	return cursplit

def augment(sample, label, weight=None, ifflip=True, ifswap=False, ifsmooth=False, ifjitter=False):
	"""
	:param sample, the cropped sample input
	:param label, the corresponding sample ground-truth
	:param weight, 
	:param ifflip, flag for random flipping
	:param ifswap, flag for random swapping
	:param ifsmooth, flag for Gaussian smoothing on the CT image
	:param ifjitter, flag for intensity jittering on the CT image
	:return: augmented training samples
	"""
	if ifswap:
		if sample.shape[0] == sample.shape[1] and sample.shape[0] == sample.shape[2]:
			axisorder = np.random.permutation(3)
			sample = np.transpose(sample, axisorder)
			label = np.transpose(label, axisorder)
			if weight is not None:
				weight = np.transpose(weight,np.concatenate([[0],axisorder+1]))
	
	# prob_aug = random.random()
	if ifflip :#and prob_aug > 0.5:
		flipid = np.random.randint(2)*2-1
		sample = np.ascontiguousarray(sample[:,:,::flipid])
		label = np.ascontiguousarray(label[:,:,::flipid])
		if weight is not None:
			weight = np.ascontiguousarray(weight[:,:,::flipid])

	prob_aug = random.random()
	if ifjitter and prob_aug > 0.5:
		ADD_INT = (np.random.rand(sample.shape[0], sample.shape[1], sample.shape[2])*2 - 1)*10
		ADD_INT = ADD_INT.astype('float')
		cury_roi = label*ADD_INT/255.0
		sample += cury_roi
		sample[sample < 0] = 0
		sample[sample > 1] = 1

	prob_aug = random.random()
	if ifsmooth and prob_aug > 0.5:
		sigma = np.random.rand()
		if sigma > 0.5:
			sample = gaussian_filter(sample, sigma=1.0)

	return sample, label, weight

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

def location_airway_sample(img, label, weight, loc, crop_size):
    origin_size = img.shape
    random_loc = np.random.randint(len(loc[0])) # (x , y , z) 其中x,t,z是列表
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop

def skeleton_sample(img, label, pred, skeleton, dist, crop_size):
    origin_size = img.shape
    crop_size = np.array(crop_size)
    for i in range(3):
        if crop_size[i] >= origin_size[i]:
            pad_num = (crop_size[i] - origin_size[i])//2 + 1
            img = np.pad(img, pad_num, 'constant')
            label = np.pad(label, pad_num, 'constant')
            pred = np.pad(pred, pad_num, 'constant')
            skeleton = np.pad(skeleton, pad_num, 'constant')
            dist = np.pad(dist, pad_num, 'constant')
    origin_size = img.shape
    if (pred*skeleton).sum() == skeleton.sum():
        start = [np.random.randint(0,origin_size[0]-crop_size[0]),np.random.randint(0,origin_size[1]-crop_size[1]),np.random.randint(0,origin_size[2]-crop_size[2])]
        img2 = img[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
        label2 = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        dist2 = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    else:
        loc = np.where(skeleton*(1-pred))
        random_loc = np.random.randint(len(loc[0]))
        start = [np.random.randint(max(0,loc[0][random_loc]-crop_size[0]), loc[0][random_loc]),
                 np.random.randint(max(0,loc[1][random_loc]-crop_size[1]), loc[1][random_loc]),
                 np.random.randint(max(0,loc[2][random_loc]-crop_size[2]), loc[2][random_loc])]
        for i in range(3):
            if (start[i]+crop_size[i]) > origin_size[i]:
                start[i] = origin_size[i] - crop_size[i] 
        img2 = img[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
        label2 = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        dist2 = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    
    return img2, label2, dist2

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
