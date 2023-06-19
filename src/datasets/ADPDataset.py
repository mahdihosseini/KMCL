import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from typing import Any


class ADPDataset(Dataset):
    db_name = 'ADP V1.0 Release'
    ROI = 'img_res_1um_bicubic'
    csv_file = 'ADP_EncodedLabels_Release1_Flat.csv'
    # Classes from https://github.com/mahdihosseini/ADP/blob/5b2508e8c4c513f8a556d57fd312a1222e2dfe77/src/htt_def.py
    # Using p classes
    # TODO add non-p classes from ADP-Release-Flat

    ADP_classes = {
        "L1": {
            "numClasses": 9,
            "classesNames": ["E", "C", "H", "S", "A", "M", "N", "G", "T"]
        },
        "L2": {
            "numClasses": 26,
            "classesNames": ["E",
                             "E.M", "E.T",
                             "C",
                             "C.D", "C.L",
                             "H",
                             "H.E", "H.K", "H.Y",
                             "S",
                             "S.M", "S.C", "S.R",
                             "A",
                             "A.W", "A.M",
                             "M",
                             "N",
                             "N.P", "N.R", "N.G",
                             "G",
                             "G.O", "G.N",
                             "T"]
        },
        "L3": {
            "numClasses": 33,
            "classesNames": ["E",
                             "E.M", "E.M.S", "E.M.C", "E.T", "E.T.S", "E.T.C",
                             "C",
                             "C.D", "C.D.I", "C.D.R", "C.L",
                             "H",
                             "H.E", "H.K", "H.Y",
                             "S",
                             "S.M", "S.C", "S.R",
                             "A",
                             "A.W", "A.M",
                             "M",
                             "N",
                             "N.P", "N.R", "N.G", "N.G.M",
                             "G",
                             "G.O", "G.N",
                             "T"]
        },
        "L3Only": {
            "numClasses": 22,
            "classesNames": ['E.M.S', 'E.M.C', 'E.T.S', 'E.T.C',
                             'C.D.I', 'C.D.R', 'C.L',
                             'H.E', 'H.K', 'H.Y',
                             'S.M', 'S.C', 'S.R',
                             'A.W', 'A.M',
                             'M',
                             'N.P', 'N.R', 'N.G.M',
                             'G.O', 'G.N',
                             'T']
        }
    }

    def __init__(self,
                 level,
                 root,
                 split='train',
                 transform=None,
                 loader=default_loader):# -> None:
        """
        Args:
            level (str): a string corresponding to a dict
                defined in code later on in this file
                defines the hierarchy to be trained on
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            root (string): Root directory of the ImageNet Dataset.
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision

        Attributes:
            self.full_image_paths (list) : a list of image paths
            self.class_labels (np.ndarray) : a numpy array of class labels
                (num_samples, num_classes)
            self.samples (list): a list of (image_path, label)
            cls.ADP_classes: a dictionary of classes for various hierarchies
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting paths:
        csv_file_path = os.path.join(self.root, self.db_name, self.csv_file)

        ADP_data = pd.read_csv(filepath_or_buffer=csv_file_path, header=0)  # reads data and returns a pd.dataframe
        # rows are integers starting from 0, columns are strings: e.g. "Patch Names", "E", ...

        split_folder = os.path.join(self.root, self.db_name, 'splits')

        if self.split == "train":
            train_inds = np.load(os.path.join(split_folder, 'train.npy'))
            out_df = ADP_data.loc[train_inds, :]

        elif self.split == "valid":
            valid_inds = np.load(os.path.join(split_folder, 'valid.npy'))
            out_df = ADP_data.loc[valid_inds, :]

        elif self.split == "test":
            test_inds = np.load(os.path.join(split_folder, 'test.npy'))
            out_df = ADP_data.loc[test_inds, :]

        self.full_image_paths = [os.path.join(self.root, self.db_name, self.ROI, image_name) for image_name in
                                 out_df['Patch Names']]
        self.class_labels = out_df[self.ADP_classes[level]['classesNames']].to_numpy(dtype=np.float32)
        
        NewImagePaths = []
        NewClassLabels = []
        print("Original", len( self.full_image_paths))
        for idx in range(len(self.class_labels)):
            if sum(self.class_labels[idx]) != 0:
                NewImagePaths.append(self.full_image_paths[idx])
                NewClassLabels.append(self.class_labels[idx])
        
        self.full_image_paths = NewImagePaths
        self.class_labels = NewClassLabels
        print("New", len( self.full_image_paths))
        # defined for compatibility with pytorch ImageFolder
        self.samples = [(self.full_image_paths[i], self.class_labels[i]) for i in range(len(self))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(out_df[self.ADP_classes[level]['classesNames']].columns)}
        # self.class_labels = np.sum(self.class_labels, axis=0)

    def __getitem__(self, idx):# -> [Any, torch.Tensor]:

        path = self.full_image_paths[idx]
        label = self.class_labels[idx]

        sample = self.loader(path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.full_image_paths)




