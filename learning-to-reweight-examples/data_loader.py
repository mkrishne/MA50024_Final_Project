import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd

class MNISTImbalanced():
    def __init__(self, n_items = 5000, classes=[9, 4], proportion=0.9, n_val=5, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('data',train=True, download=True)
        else:
            self.mnist = datasets.MNIST('data',train=False, download=True)
            proportion = 0.5
            n_val = 0
        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        n_class = [0, 0]
        n_class[0] = int(np.floor(n_items*proportion))
        n_class[1] = n_items - n_class[0]

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        if mode == "train":
            data_source = self.mnist.train_data
            label_source = self.mnist.train_labels
        else:
            data_source = self.mnist.test_data
            label_source = self.mnist.test_labels

        for i, c in enumerate(classes):
            tmp_idx = np.where(label_source == c)[0]
            np.random.shuffle(tmp_idx)
            tmp_idx = torch.from_numpy(tmp_idx)
            img = data_source[tmp_idx[:n_class[i] - n_val]]
            self.data.append(img)
            
            cl = label_source[tmp_idx[:n_class[i] - n_val]]
            self.labels.append((cl == classes[0]).float())

            if mode == "train":
                img_val = data_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                for idx in range(img_val.size(0)):
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)

                    self.data_val.append(img_tmp.unsqueeze(0))

                cl_val = label_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                self.labels_val.append((cl_val == classes[0]).float())

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        if mode == "train":
            self.data_val = torch.cat(self.data_val, dim=0)
            self.labels_val = torch.cat(self.labels_val, dim=0)





    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
 
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def get_mnist_loader(batch_size, classes=[9, 4], n_items=5000, proportion=0.9, n_val=5, mode='train'):
    """Build and return data loader."""

    dataset = MNISTImbalanced(classes=classes, n_items=n_items, proportion=proportion, n_val=n_val,mode=mode)

    shuffle = False
    if mode == 'train':
        shuffle = True
    shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

class CIFAR_FLIP_EXP():
    def __init__(self, noise_ratio=0.4, n_val=1000, random_seed=1, cifar_type="10", mode="train", run_type="unif_flip"):
        if mode == "train":
            if cifar_type == "10":
              self.cifar = datasets.CIFAR10('data',train=True, download=True)
            else:
              self.cifar = datasets.CIFAR100('data',train=True, download=True)
        else:
            if cifar_type == "10":
              self.cifar = datasets.CIFAR10('data',train=False, download=True)
            else:
              self.cifar = datasets.CIFAR100('data',train=False, download=True)
            n_val = 0
        
        num_labels = 10
        if(cifar_type == "100"):
          num_labels = 100

        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        data_source = self.cifar.data
        label_source = self.cifar.targets

        if mode == "train":
          data_val = data_source[-n_val:]
          labels_val = label_source[-n_val:]
          
          print(data_source.shape[0])
          flip_num = int(noise_ratio*data_source.shape[0])
          
          print(run_type)
          if(run_type == "unif_flip"):
            for i in range(flip_num):
              other_labels = list(range(0, label_source[i])) + list(range(label_source[i+1], num_labels))
              label_source[i] = random.choice(other_labels)
          elif(run_type == "bkgnd_flip"):
            label_source[:flip_num] = [3 for i in range(flip_num)]
          
          for idx in range(data_val.shape[0]):
            img_tmp = Image.fromarray(data_val[idx], mode='RGB')
            img_tmp = self.transform(img_tmp)
            
            self.data_val.append(img_tmp.unsqueeze(0))
            self.labels_val.append(torch.tensor(labels_val[idx]).unsqueeze(0))

          self.data_val = torch.cat(self.data_val, dim=0)
          self.labels_val = torch.cat(self.labels_val, dim=0)
          print(self.data_val.shape)
          print(self.labels_val.shape)

        self.data.append(data_source)
        self.labels.append(label_source)

        self.data = torch.tensor(self.data).squeeze(0)
        self.labels = torch.tensor(self.labels).squeeze(0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
 
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR_VAL():
    def __init__(self, data_val, labels_val):
        
        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.data_val = data_val
        self.labels_val = labels_val

    def __len__(self):
        return self.data_val.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
 
        img, target = self.data_val[index], self.labels_val[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_cifar_loader(batch_size, noise_ratio=0.4, n_val=1000, cifar_type = "10", mode='train',run_type='unif_flip"'):
    """Build and return data loader."""

    dataset = CIFAR_FLIP_EXP(noise_ratio=noise_ratio, n_val=n_val,cifar_type = cifar_type, mode=mode, run_type=run_type )

    shuffle = False
    if mode == 'train':
        shuffle = True
    shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle, num_workers=2)
    return data_loader
