import os 
from utils import load_itk_image, save_itk
from tqdm import tqdm

# 需要使用Lungmask提取训练和验证数据集的mask，从而提取小气道
def ex_small_airway():
    mask_path = r'/home/wqb/wqb/dataset/EXACT09/lungmask_clean/train'
    label_path = r'/home/wqb/wqb/dataset/EXACT09/label_clean/train'

    mask_files = os.listdir(mask_path)
    label_files = os.listdir(label_path)

    save_path = r'/home/wqb/wqb/dataset/EXACT09/smallairway_clean/train'

    for i in tqdm(range(len(label_files))):
        name = label_files[i].split('/')[-1].split('_label')[0]
        path1 = os.path.join(label_path,label_files[i])     
        print(path1)
        label , oring , spacing = load_itk_image(path1)
        path2 = path1.replace('label','lungmask')
        print(path2)
        mask ,_ , _ = load_itk_image(path2)
        sm = label * mask  
        path3 = os.path.join(save_path,name + '_smallairway.nii.gz') 
        print(path3)    
        save_itk(sm,oring,spacing,path3)

    

if __name__ == '__main__':
	ex_small_airway()

