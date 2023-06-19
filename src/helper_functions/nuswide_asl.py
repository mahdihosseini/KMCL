import os, sys
import os.path as osp
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

class NusWideAslDataset(Dataset):
    def __init__(self, 
        img_dir,
        csv_path, 
        split,
        transform=None,
         ) -> None:
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.split = split
        assert split in ['all', 'train', 'val']
        self.transform = transform

        self.itemlist = self.preprocess()

    def preprocess(self):
        # read csv file
        df = pd.read_csv(self.csv_path)
        labels_col = df['label']
        labels_list_all = []
        for item in labels_col:
            i_labellist = str_to_list(item)
            labels_list_all.extend(i_labellist)
        labels_list_all = sorted(list(set(labels_list_all)))
        labels_map = {labelname:idx for idx, labelname in enumerate(labels_list_all)}
        length = len(labels_list_all)

        # generate itemlist
        res = []
        for index, row in df.iterrows():
            split_name = row[2]
            if split_name != self.split and self.split != 'all':
                continue
            filename = row[0]
            imgpath = osp.join(self.img_dir, filename)
            label = [labels_map[i] for i in str_to_list(row[1])]
            label_np = np.zeros(length, dtype='float32')
            for idd in label:
                label_np[idd] = 1.0
            res.append((imgpath, label_np))

        return res

    def __len__(self) -> int:
        return len(self.itemlist)

    def __getitem__(self, index: int):
        imgpath, labels = self.itemlist[index]

        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, labels


def str_to_list(text):
    """
    input: "['clouds', 'sky']" (str)
    output: ['clouds', 'sky'] (list)

    """
    # res = []
    res = [i.strip('[]\'\"\n ') for i in text.split(',')]
    return res


if __name__ == '__main__':
    ds = NusWideAslDataset(
            img_dir='/data/shilong/data/nus_wide/nuswide_asl',
            csv_path='/data/shilong/data/nus_wide/nuswide_asl/nus_wid_data.csv',
            split='train',
            transform=None
        )
    print('len(ds):', len(ds))
    # print(ds[0])

    ds = NusWideAslDataset(
            img_dir='/data/shilong/data/nus_wide/nuswide_asl',
            csv_path='/data/shilong/data/nus_wide/nuswide_asl/nus_wid_data.csv',
            split='val',
            transform=None
        )
    print('len(ds):', len(ds))
    print(ds[0])

    # ds = NusWideAslDataset(
    #         img_dir='/data/shilong/data/nus_wide/nuswide_asl',
    #         csv_path='/data/shilong/data/nus_wide/nuswide_asl/nus_wid_data.csv',
    #         split='all',
    #         transform=None
    #     )
    # print('len(ds):', len(ds))