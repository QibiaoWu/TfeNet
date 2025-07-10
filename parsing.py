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

def parsing(label,origin,spacing):
    skel = skeletonize_3d(label)
    
    #separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skel, neighbor_filter) * skel
    #distribution = skeleton_filtered[skeleton_filtered>0]
    #plt.hist(distribution)
    skeleton_parse = skel.copy()
    skeleton_parse[skeleton_filtered>3] = 0
    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    #remove small branches
    # for i in range(num):
    #     a = cd[cd==(i+1)]
    #     if a.shape[0]<5:
    #         skeleton_parse[cd==(i+1)] = 0
    # cd, num = ndimage.label(skeleton_parse, structure = con_filter)


    #parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1-skeleton_parse, return_indices=True) # 距离变换
    dist, _ = ndimage.distance_transform_edt(label, return_indices=True) # 距离变换
    dist = dist[np.newaxis,np.newaxis,...]
    label_torch = torch.from_numpy(label).to(torch.float32)
    tree_parsing = np.zeros(label.shape, dtype = np.uint16)
    tree_parsing = cd[inds[0,...], inds[1,...], inds[2,...]] * label
    tree_parsing = torch.from_numpy(tree_parsing).to(torch.int32).unsqueeze(0).unsqueeze(0)
    parsing = torch.zeros_like(label_torch).unsqueeze(0).unsqueeze(0).to(torch.float32)
    for label_id in range(1, num + 1):
        # 获取当前分支的所有体素位置
        branch_voxels = torch.where(tree_parsing == label_id)
        padding_out = torch.zeros_like(label_torch).unsqueeze(0).unsqueeze(0)
        padding_out[branch_voxels] = 1

        # 计算该分支的边界体素到骨架的最大距离
        max_distance = np.max(dist[branch_voxels])
     
        if max_distance <= 4:
            parsing[branch_voxels] = 2
        else: 
            parsing[branch_voxels] = 1

    
    parsing = parsing*label_torch
    parsing = parsing.squeeze().float()

    return parsing


def save_alpha_weight(label_path, save_path):
    file_names = os.listdir(label_path)

    for name in file_names:
        print(name)
        path = os.path.join(label_path,name)
        label_img,origin,spacing = load_itk_image(path)
        label = sitk.ReadImage(path)
        label_img = sitk.GetArrayFromImage(label)
        tree_parsing = parsing(label_img,origin,spacing)
        save_name = os.path.join(save_path, name.split('label.nii')[0] + 'parsing.nii.gz')
        save_itk(tree_parsing.squeeze(),origin,spacing,save_name)
       
       

if __name__ == '__main__':
    # save_local_imbalance_based_weight(r'/home/wqb/wqb/dataset/BAS/label_clean/train',r'/home/wqb/wqb/dataset/BAS/LIB_weight/train')
    save_alpha_weight(r'/home/wqb/wqb/dataset/BAS/label_clean/train',r'/home/wqb/wqb/dataset/BAS/parsing/train')

