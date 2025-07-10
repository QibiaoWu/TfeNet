import SimpleITK as sitk
import numpy as np
import os
import skimage.measure as measure
from scipy import ndimage
import json

EPSILON = 1e-32
def compute_binary_iou(y_true, y_pred):
    intersection     = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou

def large_connected_domain(label, conn=1):
    cd, num = measure.label(label, return_num=True, connectivity=conn)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    large_cd = ndimage.binary_fill_holes(large_cd)

    # iou = compute_binary_iou(label, large_cd)
    # print(iou)

    # flag=-1
    # while iou < 0.1:
    #     print(" failed cases, require find next large connected component")
    #     large_cd = (cd == (volume_sort[flag-1] + 1)).astype(np.uint8)
    #     flag -= 1
    #     iou = compute_binary_iou(label, large_cd)

    return large_cd.astype('float')

def postprocess(root, save_root): 
    print("postprocess begin")
    name_list = os.listdir(root)
    labels_names = [file for file in name_list if '.nii' in file]
    name_list.sort()
    for label_name in labels_names:
        name = label_name.split('.nii.gz')[0]
        # pred_name = label_name.replace('-gt','-pred')
        print('post processing on:', name)
        # label = sitk.ReadImage(os.path.join(root, label_name))
        pred = sitk.ReadImage(os.path.join(root, label_name))
        pred_img= sitk.GetArrayFromImage(pred)
        pred_img = large_connected_domain(pred_img)

        pred_save = sitk.GetImageFromArray(pred_img)
        pred_save.SetOrigin(pred.GetOrigin())
        pred_save.SetDirection(pred.GetDirection())
        pred_save.SetSpacing(pred.GetSpacing())

    
        sitk.WriteImage(pred_save, os.path.join(save_root, name + '.nii.gz'))
    print("postprocess end")

def back_original_size(result_root, save_root):
    ori_root = './data/imagesVal'
    # result_root = './result/test_post'
    # save_root = './result/test_orisize'
    file_path = './data/lung_bbox_val_dict.json'
    name_list = os.listdir(result_root)
    with open(file_path, 'r') as file:
        pos_dic = json.load(file)

    name_list.sort()
    for name in name_list[:]:
        print(name)
        image = sitk.ReadImage(os.path.join(ori_root, name))
        array = sitk.GetArrayFromImage(image)
        result = np.zeros_like(array)
        pred = sitk.ReadImage(os.path.join(result_root, name))
        pred = sitk.GetArrayFromImage(pred)
        pos = pos_dic[name]
        zmin, zmax, ymin, ymax, xmin, xmax = pos
        shape = pred.shape
        result[zmin:zmin+shape[0], ymin:ymin+shape[1], xmin:xmin+shape[2]] = pred
        result_image = sitk.GetImageFromArray(result.astype(np.byte))
        result_image.SetOrigin(image.GetOrigin())
        result_image.SetDirection(image.GetDirection())
        result_image.SetSpacing(image.GetSpacing())
        sitk.WriteImage(result_image, os.path.join(save_root, name))

def check_meta(name):
    my_root = './result/test_orisize'
    root = '/home/tangwen/Documents/2022_experiment_hospital/ATM2022/temp/'
    img1 = sitk.ReadImage(os.path.join(my_root, name))
    img2 = sitk.ReadImage(os.path.join(root, name))
    print(img1.GetOrigin(), img2.GetOrigin())
    print(img1.GetDirection(), img2.GetDirection())
    print(img1.GetSpacing(), img2.GetSpacing())
    arr1 = sitk.GetArrayFromImage(img1)
    arr2 = sitk.GetArrayFromImage(img2)
    print(arr1.shape, arr2.shape)
    print(arr1.dtype, arr2.dtype)
    cd1, num1 = measure.label(arr1, return_num=True, connectivity=1)
    cd2, num2 = measure.label(arr2, return_num=True, connectivity=1)
    print(num1, num2)

def merge_multi_result(folder_names, save_folder):
    root = './result'
    file_list = os.listdir(os.path.join(root, folder_names[0]))
    file_list = [f for f in file_list if f.endswith('.npy')]
    for file in file_list:
        print(file)
        pred = np.load(os.path.join(root, folder_names[0], file))
        for name in folder_names[1:]:
            _pred = np.load(os.path.join(root, name, file))
            pred = pred + _pred
        pred = pred / (len(folder_names))
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        image = sitk.GetImageFromArray(pred.astype(np.byte))
        sitk.WriteImage(image, os.path.join(save_folder, file.replace('.npy', '.nii.gz')))


if __name__ == '__main__':
    postprocess(r'./predict_result/concat', r'./predict_result/outputs')
















