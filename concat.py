import os 
from utils import load_itk_image, save_itk
from tqdm import tqdm
import numpy as np

# 合并气道图
def concat_airway():
    small_label_path = './predict_result/pred_small'
    label_path = './predict_result/pred'

    label_files = os.listdir(label_path)

    save_path = './predict_result/concat'

    for i in tqdm(range(len(label_files))):
        print(label_files[i])
        name = label_files[i].split('/')[-1].split('.nii')[0]
        path = os.path.join(label_path,label_files[i])     
        label , oring , spacing = load_itk_image(path)
        path = os.path.join(small_label_path,label_files[i])   
        small_label ,_ , _ = load_itk_image(path)
        concat = label + small_label  
        concat[concat>0]=1
        concat = concat.astype('uint8')
        path = os.path.join(save_path,name + '.nii.gz')     
        save_itk(concat,oring,spacing,path)
    

if __name__ == '__main__':
	concat_airway()

