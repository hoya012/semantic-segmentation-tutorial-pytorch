import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.autograd import Variable

from learning.minicity import MiniCity
from learning.model import convert_bn_to_instancenorm, convert_bn_to_evonorm, convert_bn_to_groupnorm, DeepLabHead, UNet

"""
====================
Data Loader Function
====================
"""
def get_dataloader(dataset, args):
    #args = args

    def test_trans(image, mask=None):
        # Resize, 1 for Image.LANCZOS
        image = TF.resize(image, args.test_size, interpolation=1) 
        # From PIL to Tensor
        image = TF.to_tensor(image)
        # Normalize
        image = TF.normalize(image, args.dataset_mean, args.dataset_std)
        
        if mask:
            # Resize, 0 for Image.NEAREST
            mask = TF.resize(mask, args.test_size, interpolation=0) 
            mask = np.array(mask, np.uint8) # PIL Image to numpy array
            mask = torch.from_numpy(mask) # Numpy array to tensor
            return image, mask
        else:
            return image

    def train_trans(image, mask):
        # Generate random parameters for augmentation
        bf = np.random.uniform(1-args.colorjitter_factor,1+args.colorjitter_factor)
        cf = np.random.uniform(1-args.colorjitter_factor,1+args.colorjitter_factor)
        sf = np.random.uniform(1-args.colorjitter_factor,1+args.colorjitter_factor)
        hf = np.random.uniform(-args.colorjitter_factor,+args.colorjitter_factor)
        pflip = np.random.randint(0,1) > 0.5

        # Random scaling
        scale_factor = np.random.uniform(0.75, 2.0)
        scaled_train_size = [int(element * scale_factor) for element in args.train_size]

        # Resize, 1 for Image.LANCZOS
        image = TF.resize(image, scaled_train_size, interpolation=1)
        # Resize, 0 for Image.NEAREST
        mask = TF.resize(mask, scaled_train_size, interpolation=0) 
        
        # Random cropping
        if not args.train_size == args.crop_size:
            if image.size[1] <= args.crop_size[0]: # PIL image: (width, height) vs. args.size: (height, width)
                pad_h = args.crop_size[0] - image.size[1] + 1
                pad_w = args.crop_size[1] - image.size[0] + 1
                image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=19)

            # From PIL to Tensor
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            h, w = image.size()[1], image.size()[2] #scaled_train_size #args.train_size
            th, tw = args.crop_size
            
            i = np.random.randint(0, h - th)
            j = np.random.randint(0, w - tw)
            image_crop = image[:,i:i+th,j:j+tw]
            mask_crop = mask[:,i:i+th,j:j+tw]

            image = TF.to_pil_image(image_crop)
            mask = TF.to_pil_image(mask_crop[0,:,:])
        
        # H-flip
        if pflip == True and args.hflip == True:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Color jitter
        image = TF.adjust_brightness(image, bf)
        image = TF.adjust_contrast(image, cf)
        image = TF.adjust_saturation(image, sf)
        image = TF.adjust_hue(image, hf)

        # From PIL to Tensor
        image = TF.to_tensor(image)
        
        # Normalize
        image = TF.normalize(image, args.dataset_mean, args.dataset_std)
        
        # Convert ids to train_ids
        mask = np.array(mask, np.uint8) # PIL Image to numpy array
        mask = torch.from_numpy(mask) # Numpy array to tensor
            
        return image, mask

    trainset = dataset(args.dataset_path, split='train', transforms=train_trans)
    valset = dataset(args.dataset_path, split='val', transforms=test_trans)
    testset = dataset(args.dataset_path, split='test', transforms=test_trans)
    dataloaders = {}    
    dataloaders['train'] = torch.utils.data.DataLoader(trainset,
               batch_size=args.batch_size, shuffle=True,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(testset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers)

    return dataloaders

"""
====================
Focal Loss
code reference: https://github.com/clcarwin/focal_loss_pytorch
====================
"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


"""
====================
Loss Function
====================
"""

def get_lossfunc(dataset, args):
    # Define loss, optimizer and scheduler
    if args.loss == 'ce':
        if args.model == 'DeepLabv3_resnet50':
            criterion = nn.CrossEntropyLoss(ignore_index=dataset.voidClass)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.loss == 'weighted_ce':
        # Class-Weighted loss
        class_weight = [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
        class_weight.append(0) #for void-class
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=dataset.voidClass)
    elif args.loss =='focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        raise NameError('Loss is not defined!')

    return criterion


"""
====================
Model Architecture
====================
"""

def get_model(dataset, args):
    if args.model == 'UNet':
        """ U-Net baeline """
        model = UNet(len(dataset.validClasses), batchnorm=True)
    elif args.model == 'DeepLabv3_resnet50':
        """ DeepLab v3 ResNet50 """
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
        model.classifier = DeepLabHead(2048, len(dataset.validClasses))
    elif args.model == 'DeepLabv3_resnet101':
        """ DeepLab v3 ResNet101 """
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
        model.classifier = DeepLabHead(2048, len(dataset.validClasses))
    else:
        raise NameError('Model is not defined!')
    
    # Normalization Layer
    if args.norm == 'instance':
        convert_bn_to_instancenorm(model)
    elif args.norm == 'evo':
        convert_bn_to_evonorm(model)
    elif args.norm == 'group':
        convert_bn_to_groupnorm(model, num_groups=32)
    elif args.norm == 'batch':
        pass
    else:
        raise NameError('Normalization is not defined!')

    return model

"""
====================
random bbox function for cutmix
====================
"""

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

"""
====================
Custom copyblob function for copyblob data augmentation
====================
"""

def copyblob(src_img, src_mask, dst_img, dst_mask, src_class, dst_class):
    mask_hist_src, _ = np.histogram(src_mask.numpy().ravel(), len(MiniCity.validClasses)-1, [0, len(MiniCity.validClasses)-1])
    mask_hist_dst, _ = np.histogram(dst_mask.numpy().ravel(), len(MiniCity.validClasses)-1, [0, len(MiniCity.validClasses)-1])

    if mask_hist_src[src_class] != 0 and mask_hist_dst[dst_class] != 0:
        """ copy src blob and paste to any dst blob"""
        mask_y, mask_x = src_mask.size()
        """ get src object's min index"""
        src_idx = np.where(src_mask==src_class)
        
        src_idx_sum = list(src_idx[0][i] + src_idx[1][i] for i in range(len(src_idx[0])))
        src_idx_sum_min_idx = np.argmin(src_idx_sum)        
        src_idx_min = src_idx[0][src_idx_sum_min_idx], src_idx[1][src_idx_sum_min_idx]
        
        """ get dst object's random index"""
        dst_idx = np.where(dst_mask==dst_class)
        rand_idx = np.random.randint(len(dst_idx[0]))
        target_pos = dst_idx[0][rand_idx], dst_idx[1][rand_idx] 
        
        src_dst_offset = tuple(map(lambda x, y: x - y, src_idx_min, target_pos))
        dst_idx = tuple(map(lambda x, y: x - y, src_idx, src_dst_offset))
        
        for i in range(len(dst_idx[0])):
            dst_idx[0][i] = (min(dst_idx[0][i], mask_y-1))
        for i in range(len(dst_idx[1])):
            dst_idx[1][i] = (min(dst_idx[1][i], mask_x-1))
        
        dst_mask[dst_idx] = src_class
        dst_img[:, dst_idx[0], dst_idx[1]] = src_img[:, src_idx[0], src_idx[1]]