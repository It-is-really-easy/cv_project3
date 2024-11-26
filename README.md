------

# KCF目标跟踪项目

本项目基于 **Kernelized Correlation Filters (KCF)** 算法，针对单目标跟踪任务进行了优化和改进，探索了HOG特征提取、PCA降维、多尺度检测等技术的实际应用，同时对比了OpenCV的KCF实现与改进方法的性能。

## 文件结构

- **`kcf/`**
   手写的KCF实现，包含基础的HOG特征提取与目标跟踪功能。
- **`opencv_kcf/`**
   调用OpenCV自带的KCF模块，实现对目标的跟踪，作为基准对比。
- **`pca_kcf/`**
   对开源代码进行改进，添加了HOG特征的PCA降维、多尺度目标检测以及其他优化功能。

------

## 使用说明

每个文件夹都支持两种运行方式：

1. **运行指定视频文件**
2. **通过摄像头实时跟踪目标**

以下是运行步骤：

### 1. 指定视频文件

在运行时提供视频文件路径，进行目标跟踪：

```bash
python track.py --video path/to/video.mp4  
```

### 2. 实时摄像头模式

直接调用摄像头进行实时目标跟踪：

```bash
python track.py --camera  
```

### 文件夹详细说明

#### **`kcf/`**

- 实现了手写的KCF算法，包含基本的HOG特征提取与目标检测功能。

- 支持对视频和实时摄像头的目标跟踪。

- 运行方式：

  ```bash
  python kcfdet.py ./Ke.mp4
  python kcfdet.py  
  ```

#### **`opencv_kcf/`**

- 使用OpenCV的KCF模块，作为对比基准。

- 支持多种输入源，适合初学者快速理解KCF的原理。

- 运行方式：

  ```bash
  python cv2kcfdet.py ./Ke.mp4
  python cv2kcfdet.py
  ```

#### **`pca_kcf/`**

- 在基础KCF实现上增加了以下改进：

  1. **HOG特征的归一化与PCA降维**，显著降低计算复杂度，提高特征鲁棒性。
  2. **多尺度检测**，适应目标尺寸变化。
  3. **汉宁窗处理**，减少边界效应带来的影响。

- 支持多输入源和优化功能测试。

- 运行方式：

  ```bash
  python run.py  ./Ke.mp4 
  python run.py   
  ```

------

## 环境配置

### 依赖库

请先安装以下Python库：

```bash
pip install numpy opencv-python  
```

### 硬件要求

- GPU支持（推荐，但非必要）
- NVIDIA GeForce RTX 3070Ti或更高版本（建议）

------

## 结果对比

运行完成后，可以通过文件夹中的`mine.mp4`和`opencv.mp4`对比改进前后的效果：

- 改进版的KCF（pca_kcf）在光照变化、背景噪声、目标尺寸变化等条件下表现更稳定。
- OpenCV自带的KCF在复杂场景下可能丢失目标。

------

## 参考链接

1. [Kernelized Correlation Filter GitHub项目](https://github.com/LiangshouX/Kernelized-Correlation-Filter-KCF-/tree/master/data)
2. [改进版代码来源](https://github.com/uoip/KCFpy)
3. [汉宁窗处理相关资料](https://zhuanlan.zhihu.com/p/340686487)

已上传[It-is-really-easy/cv_project3](https://github.com/It-is-really-easy/cv_project3)