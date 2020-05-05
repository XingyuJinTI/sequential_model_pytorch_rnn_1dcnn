from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch.utils.data as data

import os
import os.path
import re
import torch
import pickle


def default_loader(path):
    with open(path, 'rb') as fp:
        lm_list = pickle.load(fp)
    fp.close()
    return lm_list

def default_list_reader(fileList):
    lmList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            lmPath=line.strip()[:-2].strip()
            label=line.strip()[-1]

            lmList.append((lmPath, int(label)))
    return lmList


class LandmarkList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.lmList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        lmPath, target = self.lmList[index]
        lm = self.loader(os.path.join(self.root, lmPath))
        if self.transform is not None:
            lm = F.normalize(lm, dim=0)
        return lm, target, lm.shape[0]

    def __len__(self):
        return len(self.lmList)


class LandmarkListTest(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.lmList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        lmPath, target = self.lmList[index]
        lm = self.loader(os.path.join(self.root, lmPath))
        if self.transform is not None:
            lm = F.normalize(lm, dim=0)
        return lm, target, lm.shape[0], lmPath

    def __len__(self):
        return len(self.lmList)