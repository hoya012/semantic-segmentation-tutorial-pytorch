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
   - move dataset into `minicity` folder.

### 0-3. Dataset Simple EDA (Exploratory Data Analysis) - Class Distribution, Sample Distribution
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
![](https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/minicity/class_pixel_distribution.png)

#### deeplab v3 baseline test set result
- Dataset has severe Class-Imbalance problem.  
    - IoU of minor class is very low. (wall, fence, bus, train)

```python
classes          IoU      nIoU
--------------------------------
road          : 0.963      nan
sidewalk      : 0.762      nan
building      : 0.856      nan
wall          : 0.120      nan
fence         : 0.334      nan
pole          : 0.488      nan
traffic light : 0.563      nan
traffic sign  : 0.631      nan
vegetation    : 0.884      nan
terrain       : 0.538      nan
sky           : 0.901      nan
person        : 0.732    0.529
rider         : 0.374    0.296
car           : 0.897    0.822
truck         : 0.444    0.218
bus           : 0.244    0.116
train         : 0.033    0.006
motorcycle    : 0.492    0.240
bicycle       : 0.638    0.439
--------------------------------
Score Average : 0.573    0.333
--------------------------------
```

## 1. Training Baseline Model 
- I use [DeepLabV3 from torchvision.](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
    - ResNet-50 Backbone, ResNet-101 Backbone

- I use 4 RTX 2080 Ti GPUs. (11GB x 4)
- If you have just 1 GPU or small GPU Memory, please use smaller batch size (<= 8)

```python
python baseline.py --save_path baseline_run_deeplabv3_resnet50 --crop_size 576 1152 --batch_size 8;
```

```python
python baseline.py --save_path baseline_run_deeplabv3_resnet101 --model DeepLabv3_resnet101 --train_size 512 1024 --test_size 512 1024 --crop_size 384 768 --batch_size 8;
```
 
### 1-1. Loss Functions
- I tried 3 loss functions. 
    - Cross-Entropy Loss
    - Class-Weighted Cross Entropy Loss
    - Focal Loss
- You can choose loss function using `--loss` argument.
    - I recommend default (ce) or Class-Weighted CE loss. Focal loss didn'y work well in my codebase.

```python
# Cross Entropy Loss
python baseline.py --save_path baseline_run_deeplabv3_resnet50 --crop_size 576 1152 --batch_size 8;
```

```python
# Weighted Cross Entropy Loss
python baseline.py --save_path baseline_run_deeplabv3_resnet50_wce --crop_size 576 1152 --batch_size 8 --loss weighted_ce;
```

```python
# Focal Loss
python baseline.py --save_path baseline_run_deeplabv3_resnet50_focal --crop_size 576 1152 --batch_size 8 --loss focal --focal_gamma 2.0;
```

### 1-2. Normalization Layer
- I tried 4 normalization layer.
    - Batch Normalization (BN)
    - Instance Normalization (IN)
    - Group Normalization (GN)
    - Evolving Normalization (EvoNorm)

- You can choose normalization layer using `--norm` argument.
    - I recommend BN. 

```python
# Batch Normalization
python baseline.py --save_path baseline_run_deeplabv3_resnet50 --crop_size 576 1152 --batch_size 8;
```

```python
# Instance Normalization
python baseline.py --save_path baseline_run_deeplabv3_resnet50_instancenorm --crop_size 576 1152 --batch_size 8 --norm instance;
```

```python
# Group Normalization
python baseline.py --save_path baseline_run_deeplabv3_resnet50_groupnorm --crop_size 576 1152 --batch_size 8 --norm group;
```

```python
# Evolving Normalization
python baseline.py --save_path baseline_run_deeplabv3_resnet50_evonorm --crop_size 576 1152 --batch_size 8 --norm evo;
```

### 1-3. Additional Augmentation Tricks
- Propose 2 data augmentation techniques (CutMix, copyblob)
- CutMix Augmentation
![](https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/minicity/cutmix.PNG)
   - Based on [Original CutMix](https://arxiv.org/abs/1905.04899), bring idea to Semantic Segmentation. 

- CopyBlob Augmentation
![](https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/minicity/copyblob.PNG)
   - To tackle Class-Imbalance, use CopyBlob augmentation with visual inductive prior.
      - Wall must be located on the sidewalk
      - Fence must be located on the sidewalk
      - Bus must be located on the Road
      - Train must be located on the Road

```python
# CutMix Augmentation
python baseline.py --save_path baseline_run_deeplabv3_resnet50_cutmix --crop_size 576 1152 --batch_size 8 --cutmix;
```

```python
# CopyBlob Augmentation
python baseline.py --save_path baseline_run_deeplabv3_resnet50_copyblob --crop_size 576 1152 --batch_size 8 --copyblob;
```

## 2. Inference
- After training, we can evaluate using trained models.
   - I recommend same value for `train_size` and `test_size`.

```python
python baseline.py --save_path baseline_run_deeplabv3_resnet50 --batch_size 4 --predict;
```

### 2-1. Multi-Scale Infernece (Test Time Augmentation)
- I use [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2] scales for Multi-Scale Inference. Additionaly, use H-Flip.
    - Must use single batch (batch_size=1)

```python
# Multi-Scale Inference
python baseline.py --save_path baseline_run_deeplabv3_resnet50 --batch_size 1 --predict --mst;
```

### 2-2. Calculate Metric using Validation Set
- We can calculate metric and save results into `results.txt`.
   - ex) [My final validation set result](https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/results.txt)

```python
python evaluate.py --results baseline_run_deeplabv3_resnet50/results_val --batch_size 1 --predict --mst;
```

## 3. Final Result
- ![](https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/minicity/leaderboard.PNG)
- My final single model result is **0.6069831962012341**
    - Achieve 5th place on the leaderboard.
    - But, didn't submit short-paper, so my score is not official score.
- If i use bigger model and bigger backbone, performance will be improved.. maybe..
- If i use ensemble various models, performance will be improved!
- Leader board can be found in [Codalab Challenge Page](https://competitions.codalab.org/competitions/23712#results)

## 4. Reference
- [vipriors-challange-toolkit](https://github.com/VIPriors/vipriors-challenges-toolkit)
- [torchvision deeplab v3 model](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py)
- [Focal Loss](https://github.com/clcarwin/focal_loss_pytorch)
- [Class Weighted CE Loss](https://github.com/openseg-group/OCNet.pytorch/blob/master/utils/loss.py)
- [EvoNorm](https://github.com/digantamisra98/EvoNorm)
- [CutMix Augmentation](https://github.com/clovaai/CutMix-PyTorch)