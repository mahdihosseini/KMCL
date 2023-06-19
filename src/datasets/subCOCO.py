
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
import random
import torchvision.transforms as transforms
import os
from randaugment import RandAugment
from numpy import savetxt

from numpy import loadtxt
# from src.helper_functions.helper_functions import CutoutPIL
# num_samples = 40000
class CocoSubDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None,num_samples=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        if 'train' in annFile:
            
            # random.shuffle(self.ids)
            bp = "/fs2/comm/kpgrp/mhosseini/github/ASL_Framework/ASL_reproduce/src/datasets/"
            self.ids = loadtxt(bp+'COCO_Subset_{}_Ids.csv'.format(num_samples), delimiter=',').astype(int)# self.ids[:num_samples]
            # savetxt('COCO_Subset_{}_Ids.csv'.format(num_samples), self.ids, delimiter=',')
            # print(self.ids)
            
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        self.class_labels = [i for i in range(80)]
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        
        img_id = self.ids[index]
        # print("beg", index, img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        # print(target)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        # print(len(list(coco.imgs.keys())))
        # if img_id in list(coco.imgs.keys()):
        #     print("Yes encapsulated")
        # # print()
        path = coco.imgs[img_id]['file_name']
        # import ipdb; ipdb.set_trace()
        path_list = path.split('_')
        path = os.path.join(path_list[1], path)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# data_folder = "/fs2/comm/kpgrp/mhosseini/project_MCL/"
# COCO_image_normalization_mean=[0.485, 0.456, 0.406]
# COCO_image_normalization_std=[0.229, 0.224, 0.225]
# normalize = transforms.Normalize(mean=COCO_image_normalization_mean,
#                                         std=COCO_image_normalization_std)

# instances_path_val = os.path.join(data_folder, 'coco/data/annotations/instances_val2014.json')
# instances_path_train = os.path.join(data_folder, 'coco/data/annotations/instances_train2014.json')
# data_path_val   = f'{data_folder}coco/data/'    # args.data
# data_path_train = f'{data_folder}coco/data'  # args.data
# val_dataset = CocoDetection(data_path_val,
#                             instances_path_val,
#                             transforms.Compose([
#                                 transforms.Resize((224, 224)),
#                                 transforms.ToTensor(),
#                                 normalize # no need, toTensor does normalization
#                             ]))
# train_dataset = CocoDetection(data_path_train,
#                             instances_path_train,
#                             transforms.Compose([
#                                 transforms.Resize((224, 224)),
#                                 # CutoutPIL(cutout_factor=0.5),
#                                 RandAugment(),
#                                 transforms.ToTensor(),
#                                 normalize
#                             ]))

# print(len(val_dataset))
# print(len(train_dataset))

# zeros = torch.zeros((80))
# for i,j in train_dataset:
#     label = j.max(dim=0)[0]
#     # print(label)
#     zeros += label
    
# print(zeros)
# savetxt('COCO_Subset_{}_Distribution.csv'.format(num_samples), zeros.numpy(), delimiter=',')
# Id_counts={}
# for k in range(80):
#     Id_counts[k]=0
# for i in (range(len(train_dataset))):
#     train_image, train_label = train_dataset[i]
#     bounding_boxes = train_label[:, :4]
#     class_ids = train_label[:, 4:5]
#     for j in range(80):
#         if j in class_ids:
#             Id_counts[j]+=1
# print(Id_counts)

# # We will keep retaining_percentage% total images containing required objects.
# retaining_percentage=1.25
# # We will use Id_counts_mod to keep track count of images containing each object
# Id_counts_mod={}
# for k in range(80):
#     Id_counts_mod[k]=0
    
# ImageSet = []
# LabelSet = []
# k=0
# for i in (range(len(train_dataset))):
#     train_image, train_label = train_dataset[i]
#     # bounding_boxes = train_label[:, :4]
#     class_ids = (train_label.max(dim=0)[0])#.float()train_label[:, 4:5]
#     include = False
#     #You can include/exclude any objects by changing below array.
#     for j in [i for i in range(80)]:
#         if j!=0:
#             if j in class_ids:
#                 if(Id_counts_mod[j]<Id_counts[j]*(retaining_percentage/100)):
#                     include=True
#                     break
#         else:
#             if j in class_ids:
#                 if(Id_counts_mod[j]<Id_counts[j]*(retaining_percentage/100)):
#                     for g in [1,2,3,5,7]:
#                         if g in class_ids:
#                             include=True
#                             break
#                     if(include):
#                         break
#     if include:
#         ImageSet.append(train_image)
#         LabelSet.append(train_label)
# print(train_label)
# print(len(train_image))