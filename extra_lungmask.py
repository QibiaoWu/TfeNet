from lungmask import LMInferer
import SimpleITK as sitk
import os 
from utils import load_itk_image, save_itk
from tqdm import tqdm


def ex_lungmask():
    img_path = r'/home/wqb/wqb/dataset/EXACT09/image/train'

    img_files = os.listdir(img_path)

    save_path = r'/home/wqb/wqb/dataset/EXACT09/lungmask/train'

    for i in tqdm(range(len(img_files))):
        name = img_files[i].split('/')[-1].split('.nii')[0]
        path = os.path.join(img_path,img_files[i])     
        img , oring , spacing = load_itk_image(path)

        inferer = LMInferer()
        lungmask = inferer.apply(img)  # default model is U-net(R231)

        path = os.path.join(save_path,name + '_lungmask.nii.gz')     
        print(path)
        save_itk(lungmask,oring,spacing,path)

    

if __name__ == '__main__':
	ex_lungmask()


