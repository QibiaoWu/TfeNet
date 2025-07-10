# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:23:00 2021

@author: Hao Zheng
"""
import numpy as np
import os
from scipy import ndimage
from utils import load_itk_image,save_itk
from models.LSD import LSD
import torch
import SimpleITK as sitk
from skimage.morphology import skeletonize_3d
import skimage.measure as measure

def neighbor_descriptor(label, filters):
    den = filters.sum()
    conv_label = ndimage.convolve(label.astype(np.float32), filters, mode='mirror')/den
    conv_label[conv_label==0] = 1
    conv_label = -np.log10(conv_label)
    return conv_label

def save_local_imbalance_based_weight(label_path, save_path):
    file_list = os.listdir(label_path)
    file_list.sort()
    filter0 = np.ones([7,7,7], dtype=np.float32)
    for i in range(len(file_list)):
        label,_,_ = load_itk_image(os.path.join(label_path, file_list[i])) #load the binary labels
        weight = neighbor_descriptor(label, filter0)  
        weight[weight>1]=1.     
        # weight = weight*label
        #Here is constant weight. During training, varied weighted training is adopted.
        #weight = weight**np.random.random(2,3) * label + (1-label) in dataloader.
        # weight = weight**2.5 
        weight = weight.astype(np.float32)
        save_name = os.path.join(save_path,file_list[i].split('_label')[0] + "_weight.npy")
        np.save(save_name, weight) 
        print(file_list[i])


if __name__ == '__main__':
    save_local_imbalance_based_weight(r'/home/wqb/wqb/dataset/ATM22/label_clean/train',r'/home/wqb/wqb/dataset/ATM22/LIB_weight/train')
  