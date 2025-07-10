# DSConv
该项目是一个 Pytorch C++ and CUDA Extension,采用C++和Cuda实现了DSConv的forward function和backward function,并在Python中对其进行了包装。
<br />This Project is a Pytorch C++ and CUDA Extension, which implements  the forward function and backward function for DSConv, then encapsulates C++ and CUDA  code into Python Package.

### 安装 Install
* run `python setup.py install`

#### 要求 Requires
* Python 3
* Pytorch 2.4.1


#### 速度优化  Speed Optimization
打开`src/config.h`，用户可根据自身显卡情况，设置以下两个变量，获得更快运行速度，然后运行 `python setup.py install`
<br>Unzip the downloaded compressed file, `cd modulated-deform-conv`, then open `src/config.h`,users are recommended to set the following `VARIABLES` to optimize run speed according to their NVIDIA GPU condition, then run `python setup.py install`
	* `const int CUDA_NUM_THREADS`
	* `const int MAX_GRID_NUM`

* 运行时可以通过传递`in_step`参数来优化速度，该变量控制每次并行处理的batch 大小。
<br> Or users can set different `in_step`  value in run time, which controls the batch size of each parallel processing .

## Author
**Qibiao Wu** `qibiaowu1116@163.com`
+ [github/QibiaoWU](https://github.com/QibiaoWu)


## Paper
Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation 1 Yaolei Qi, Yuting He, Xiaoming Qi, Yuan Zhang, Guanyu Yang* Southeast University
IEEE/CVF Conference on International Conference on Computer Vision 2023

## Citation
@InProceedings{Qi_2023_ICCV,
  author = {Qi, Yaolei and He, Yuting and Qi, Xiaoming and Zhang, Yuan and Yang, Guanyu},
  title = {Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2023},
  pages = {6070-6079}
}

## License
Copyright (c) 2025 Qibiao Wu
Released under the MIT license
