# Semantic Segmentation Tutorial using PyTorch
Semantic Segmentation Tutorial using PyTorch. Based on [2020 ECCV VIPriors Challange Start Code](https://github.com/VIPriors/vipriors-challenges-toolkit/tree/master/semantic-segmentation), implements semantic segmentation codebase and add some tricks.

*Editer: Hoseong Lee (hoya012)*

## 0. Experimental Setup

### 0-1. Prepare Library
```python
pip install -r requirements.txt
```

### 0-2. Download dataset (MiniCity from CityScapes)
We will use MiniCity Dataset from Cityscapes. This dataset is used for 2020 ECCV VIPriors Challenge.
- workshop page: https://vipriors.github.io/challenges/
- challenge link: https://competitions.codalab.org/competitions/23712
- [dataset download(google drive)](https://drive.google.com/file/d/1YjkiaLqU1l9jVCVslrZpip4YsCHHlbNA/view?usp=sharing)



### TO DO List
#### 1. Pre-Processing
- Augmentation 기법 조사 및 추가
   - SinGAN (https://github.com/tamarott/SinGAN)
   - GauGAN (https://github.com/NVlabs/SPADE)
   
#### 2. Post-Processing
- I want to use dense CRF. Reference:(https://github.com/lucasb-eyer/pydensecrf)

#### 3. Network & Learning 
- Class-Imbalance를 위한 여러 시도 (Batch Sampling, Multi-label stratification folding, Loss ?)
   - focal-tversky-loss(https://github.com/nabsabraham/focal-tversky-unet)
- More Network Try
   - Attention U-Net (https://github.com/LeeJunHyun/Image_Segmentation)
   - ACFNet (https://github.com/zrl4836/ACFNet/blob/master/networks/acfnet.py)
   
#### 4. Visualization & Code Implementation
- 코드 ReFactoring (지금은 baseline.py 에 함수가 덕지덕지 붙어있음 ㅠ)
   - 쪼동! 도와죠... @davinovation
- Experimental Materials Managing Tool
   - wandb (https://www.wandb.com/)
- Tensorboard 추가 


### Done
- 1. Dataset EDA (Class Distribution, Sample Distribution, etc.)

#### original class 
```python
classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
```

#### benchmark class 
```python
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
```

#### from 0 to 18 class, count labeled pixels
![](https://github.com/hoya012/viprior_challenge/blob/master/semantic-segmentation/minicity/class_pixel_distribution.png)

#### deeplab v3 baseline test set result
```python
classes          IoU      nIoU
--------------------------------
road          : 0.963      nan
sidewalk      : 0.735      nan
building      : 0.864      nan
wall          : 0.285      nan
fence         : 0.301      nan
pole          : 0.418      nan
traffic light : 0.413      nan
traffic sign  : 0.516      nan
vegetation    : 0.882      nan
terrain       : 0.453      nan
sky           : 0.895      nan
person        : 0.677    0.442
rider         : 0.295    0.164
car           : 0.882    0.746
truck         : 0.443    0.285
bus           : 0.160    0.048
train         : 0.210    0.166
motorcycle    : 0.371    0.199
bicycle       : 0.612    0.369
--------------------------------
Score Average : 0.546    0.302
--------------------------------
```

- DeepLabV3 backbone
    - I import deeplabv3_resnet101 from torch hub (https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- Loss function
    - I used IoU Loss(LovaszSoftmax Loss) from (https://github.com/bermanmaxim/LovaszSoftmax)
        - 효과는 미미했다! (0.576869276 --> 0.5477229345, val 기준)
    - Class Weighted Cross Entropy Loss (https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10)
       - 효과는 미미했다! (0.576869276 --> 0.5220565054, val 기준)
   
 
