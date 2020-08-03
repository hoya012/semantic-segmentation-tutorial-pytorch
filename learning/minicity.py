# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
from collections import namedtuple
from torchvision.datasets import Cityscapes

class MiniCity(Cityscapes):
    
    voidClass = 19
    
    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # Convert train_ids to ids
    trainid2id = np.zeros((256), dtype='uint8')
    for label in Cityscapes.classes:
        if label.train_id >= 0 and label.train_id < 255:
            trainid2id[label.train_id] = label.id
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')
    
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, 'gtFine', split)
        self.split = split
        self.images = []
        self.targets = []

        assert split in ['train','val','test'], 'Unknown value {} for argument split.'.format(split)

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            if split != 'test':
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 'gtFine_labelIds.png')
                self.targets.append(os.path.join(self.targets_dir, target_name))
            
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """

        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')
        
        if self.split != 'test':
            target = Image.open(self.targets[index])

        if self.transforms is not None:
            if self.split != 'test':
                image, target = self.transforms(image, mask=target)
                # Convert class ids to train_ids and then to tensor
                target = self.id2trainid[target]
                return image, target, filepath
            else:
                image = self.transforms(image)
                return image, filepath