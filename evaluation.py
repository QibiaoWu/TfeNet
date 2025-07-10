import os
import time
import numpy as np
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from data_ATM22 import SegValData
import skimage.measure as measure
from skimage.morphology import skeletonize_3d
from utils import load_itk_image,save_itk
import tqdm

def network_prediction(data_path, save_path,ifsmall=False):
    casemodel = import_module('TfeNet')
    config2, case_net = casemodel.get_model()
    if ifsmall:
        checkpoint = torch.load('./checkpoint/ATM22/TfeNetSmall_checkpoint.ckpt')
    else:
        checkpoint = torch.load('./checkpoint/ATM22/TfeNet_checkpoint.ckpt')
    case_net.load_state_dict(checkpoint['state_dict'])
    val_path = data_path
    dataset = SegValData(val_path)
    val_loader_case = DataLoader(dataset, batch_size=1, shuffle=False)
    case_net = case_net.cuda()
    case_net.eval()
    save_path = save_path
    # sliding window
    cube_size = 128
    step = 64
    for i, (x, origin, spacing, name) in enumerate(val_loader_case):
        case_name = name[0]
        print(case_name)
        pred = np.zeros(x.shape)
        pred_num = np.zeros(x.shape)
        # print(x.shape)
        x = x.cuda()
        xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
            (x.shape[2] - cube_size) // step + 2
        ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
            (x.shape[3] - cube_size) // step + 2
        znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 else \
            (x.shape[4] - cube_size) // step + 2
        for xx in range(xnum):
            xl = step * xx
            xr = step * xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2] - cube_size
            for yy in range(ynum):
                yl = step * yy
                yr = step * yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                for zz in range(znum):
                    zl = step * zz
                    zr = step * zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size

                    x_input = x[:, :, xl:xr, yl:yr, zl:zr]
                    p = case_net(x_input.contiguous())
                    p = p.cpu().detach().numpy()
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = np.squeeze(pred)

        print(os.path.join(save_path,case_name))
        save_itk(pred,origin[0],spacing[0],os.path.join(save_path,case_name))

if __name__ == '__main__':
    # data_path = "/your/inputs"
    data_path = "/home/wqb/wqb/dataset/BAS/image_clean/train"
    small_save_path = "./predict_result/pred_small"
    save_path = "./predict_result/pred"

    # data_path = "/home/wqb/wqb/ATM_docker_inputs"
    # small_save_path = "/home/wqb/wqb/ATM_docker_outputs/small"
    # save_path = "/home/wqb/wqb/ATM_docker_outputs/norm"

    
    network_prediction(data_path, small_save_path, ifsmall=True)
    network_prediction(data_path, save_path, ifsmall=False)
    



