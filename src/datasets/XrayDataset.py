import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
import os
import random
import pickle

disease_categories = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 2,
        'Infiltration': 3,
        'Mass': 4,
        'Nodule': 5,
        'Pneumonia': 6,
        'Pneumothorax': 7,
# }
        'Consolidation': 8,
        'Edema': 9,
        'Emphysema': 10,
        'Fibrosis': 11,
        'Pleural_Thickening': 12,
        'Hernia': 13,
        }

class CXRDataset(Dataset):

    def __init__(self, root_dir, dataset_type = 'train', transform = None):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        
        self.image_dir = os.path.join(root_dir, 'images/images')
        # self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        self.transform = transform
        self.index_dir = os.path.join(root_dir, dataset_type+'_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).iloc[0, :].to_numpy()[1:15]
        # print(self.classes)
        self.label_index = pd.read_csv(self.index_dir, header=0)
        print(len(self.label_index))
        self.label_index = self.label_index.loc[(self.label_index[list(disease_categories.keys())]!=0 ).any(axis=1)]
        print(len(self.label_index))
        # self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)

        # with open('/fs2/comm/kpgrp/mhosseini/github/MSRN/data/Xray/xray_glove_word2vec.pkl', 'rb') as f:
        #     self.inp = pickle.load(f)
        # #temp = np.array([np.array(xi) for xi in self.inp])
        # self.inp = torch.tensor(self.inp)
        self.channelExpand = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        self.class_labels = np.sum(self.label_index.to_numpy()[:,1:15].astype('int'), axis=0)
    def __len__(self):
        return int(len(self.label_index)*0.1)

    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('L')
        
        if self.transform:
            
            arr = np.expand_dims(np.array(image), -1)
            # print(arr.shape)
            arr = np.repeat(arr, 3, axis=-1)
            # print(arr.shape)
            # a = image.getdata()
            # image2 = Image.merge("RGB", [a, a, a])
            # print(arr.shape)
            
            image2 = Image.fromarray(arr)
            # print(image2.mode)
            image = self.transform(image2)
            # image = self.channelExpand(image)
            
            # print(image.shape)
        label = self.label_index.iloc[idx, 1:15].to_numpy().astype('int')
        
        # # bbox
        # bbox = np.zeros([8, 512, 512])
        # bbox_valid = np.zeros(14)
        # for i in range(8):
        #     if label[i] == 0:
        #        bbox_valid[i] = 1
        
        # cols = self.bbox_index.loc[self.bbox_index['Image Index']==name]
        # if len(cols)>0:
        #     for i in range(len(cols)):
        #         bbox[
        #             disease_categories[cols.iloc[i, 1]], #index
        #             int(cols.iloc[i, 3]/2): int(cols.iloc[i, 3]/2+cols.iloc[i, 5]/2), #y:y+h
        #             int(cols.iloc[i, 2]/2): int(cols.iloc[i, 2]/2+cols.iloc[i, 4]/2) #x:x+w
        #         ] = 1
        #         bbox_valid[disease_categories[cols.iloc[i, 1]]] = 1
        
        return image, torch.tensor(label), img_dir#, name, bbox, bbox_valid
    
    
class CXRDataset_BBox_only(Dataset):

    def __init__(self, root_dir, transform = None, data_arg=True): 
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.index_dir = os.path.join(root_dir, 'test'+'_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).ix[0, :].as_matrix()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        self.data_arg = data_arg
        
        # with open('/fs2/comm/kpgrp/mhosseini/github/MSRN/data/ADP/L1_ADP_glove_word2vec.pkl', 'rb') as f:
            # self.inp = pickle.load(f)
        #temp = np.array([np.array(xi) for xi in self.inp])
        # self.inp = torch.tensor(self.inp)
        
    def __len__(self):
        return len(self.bbox_index)

    def __getitem__(self, idx):
        name = self.bbox_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('L')
        label = self.label_index.loc[self.label_index['FileName']==name].iloc[0, 1:9].as_matrix().astype('int')
            
        
        # bbox
        bbox = np.zeros([8, 512, 512])
        bbox_valid = np.zeros(8)
        for i in range(8):
            if label[i] == 0:
               bbox_valid[i] = 1
        
        cols = self.bbox_index.loc[self.bbox_index['Image Index']==name]
        if len(cols)>0:
            for i in range(len(cols)):
                bbox[
                    disease_categories[cols.iloc[i, 1]], #index
                    int(cols.iloc[i, 3]/2): int(cols.iloc[i, 3]/2+cols.iloc[i, 5]/2), #y:y+h
                    int(cols.iloc[i, 2]/2): int(cols.iloc[i, 2]/2+cols.iloc[i, 4]/2) #x:x+w
                ] = 1
                bbox_valid[disease_categories[cols.iloc[i, 1]]] = 1
        
        #argumentation
        if self.data_arg:
            image = F.resize(image, 600, Image.BILINEAR)
            angle = random.uniform(-20, 20)
            image = F.rotate(image, angle, resample=False, expand=False, center=None)
            crop_i = random.randint(0, 600 - 512)
            crop_j = random.randint(0, 600 - 512)
            image = F.crop(image, crop_i, crop_j, 512, 512)
            bbox_list = []
            for i in range(8):
                bbox_img = Image.fromarray(bbox[i])
                bbox_img = F.resize(bbox_img, 600, Image.BILINEAR)
                bbox_img = F.rotate(bbox_img, angle, resample=False, expand=False, center=None)
                bbox_img = F.crop(bbox_img, crop_i, crop_j, 512, 512)
                bbox_img = transforms.ToTensor()(bbox_img)
                bbox_list.append(bbox_img)
            bbox = torch.cat(bbox_list)
        
        if self.transform:
            image = self.transform(image)
        
        return (image,img_dir, self.inp), torch.tensor(label)#, name, bbox, bbox_valid