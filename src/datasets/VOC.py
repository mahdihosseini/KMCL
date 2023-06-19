import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import random
import numpy as np
import torch
from PIL import Image
import pandas as pd
# import util
# from util import *
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, random_split

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                # -1:negative, 1:positive, 0:difficult --> change into 1:positive, 0:negative
                # print("B4", labels)
                # labels = (labels >= 0).astype(float)
                # print("After", labels)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images

class VOC2007(Dataset):
    def __init__(self, root, split="train", transform=None,loader=default_loader):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.transform = transform
        if split == "train":
            self.split = "trainval"
        elif split == "val" or split == "test":
            self.split = "test"
        else:
            print("dataset loading failed")
        # download dataset
        #download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + self.split + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        #self.class_labels = object_categories
        self.images = read_object_labels_csv(file_csv)
        
        # print(self.images)
        # split_index = int(len(self.images) * 0.8)

        # # Use random.sample() to sample without replacement
        # training_images = random.sample(self.images, split_index)
        # validation_images = [item for item in self.images if item not in training_images]
        # # training_images, validation_images = random_split(self.images, [int(0.8*len(self.images)), len(self.images) - int(0.8*len(self.images))])
        
        # torch.save(training_images, "VOC_Train_HP.pt")
        # torch.save(validation_images, "VOC_Val_HP.pt")
        
        # return
        # image_path = os.path.join(self.root, 'VOC2007', 'JPEGImages/')
        # self.full_image_paths = [os.path.join(image_path, image_name+'.jpg') for image_name in self.images]
        # self.class_labels=np.array([np.array(xi[1]) for xi in self.images])
         # get paths for images and label files
        image_path = os.path.join("/fs2/comm/kpgrp/mhosseini/project_MCL/voc/VOCdevkit", 'VOC2007', 'JPEGImages/')
        file_path = os.path.join("/fs2/comm/kpgrp/mhosseini/project_MCL/voc/VOCdevkit", 'VOC2007', 'ImageSets/Main/')

        # get all the images in specified dataset split
        df = pd.read_csv(os.path.join(file_path, self.split+'.txt'), header=None, names=['image'], dtype=str)
       
        # get image labels by class
        for c in object_categories:
            label_file = c + '_' + self.split + '.txt'
            dtypes = {'image': str, c: int}
            temp_df = pd.read_csv(os.path.join(file_path, label_file), header=None, names=['image', c],
                                  delim_whitespace=True, dtype=dtypes)
            df = pd.merge(df, temp_df, on='image', how='outer')

        # -1:negative, 1:positive, 0:difficult --> change into 1:positive, 0:negative
        df = df.replace([0, -1], [1, 0])
        df = df.set_index('image')
        self.class_labels = df[df.columns.tolist()].to_numpy(dtype=np.float32)
        self.full_image_paths = [os.path.join(image_path, image_name+'.jpg') for image_name in df.index]
        # self.class_labels = np.sum((self.class_labels>=0), axis=0)
        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(object_categories), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # target = (target >= 0).float()
        return img, target#torch.tensor(target)
   
    def __len__(self):
        return len(self.images)





# class HP_Tuning_VOC2007(Dataset):
#     def __init__(self, root, split="train", transform=None,loader=default_loader):
#         self.root = root
#         self.path_devkit = os.path.join(root, 'VOCdevkit')
#         self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
#         self.transform = transform
#         if split == "train":
#             self.split = "trainval"
#         elif split == "val" or split == "test":
#             self.split = "test"
#         else:
#             print("dataset loading failed")
#         # download dataset
#         #download_voc2007(self.root)

#         # define path of csv file
#         path_csv = os.path.join(self.root, 'files', 'VOC2007')
#         # define filename of csv file
#         file_csv = os.path.join(path_csv, 'classification_' + self.split + '.csv')

#         # # create the csv file if necessary
#         # if not os.path.exists(file_csv):
#         #     if not os.path.exists(path_csv):  # create dir if necessary
#         #         os.makedirs(path_csv)
#         #     # generate csv file
#         #     labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
#         #     # write csv file
#         #     write_object_labels_csv(file_csv, labeled_data)

#         #self.class_labels = object_categories
#         if split == 'train':
#             self.images = torch.load("VOC_Train_HP.pt")
#         else:
#             self.images = torch.load("VOC_Val_HP.pt")
            
#         # training_images, validation_images = random_split(self.images, [int(0.8*len(self.images)), len(self.images) - int(0.8*len(self.images))])
        
#         # np.save(training_images, "VOC_Train_HP.npy")
#         # np.save(validation_images, "VOC_Val_HP.npy")
#         # print(self.images)
#         self.class_labels=np.array([np.array(xi[1]) for xi in self.images])
#         self.class_labels = np.sum(self.class_labels, axis=0)
#         print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
#             set, len(object_categories), len(self.images)))

#     def __getitem__(self, index):
#         path, target = self.images[index]
#         img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         # target = (target >= 0).float()
#         return img, torch.tensor(target)
   
#     def __len__(self):
#         return len(self.images)
