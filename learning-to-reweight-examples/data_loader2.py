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
from sklearn.utils import shuffle

class MNIST_UNIFFLIP():
  def __init__(self, noise_ratio=0.4, n_val_per_class=10, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('data',train=True, download=True)
        else:
            self.mnist = datasets.MNIST('data',train=False, download=True)
            n_val = 0
        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        
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
        data_source = data_source[:50000]
        label_source = label_source[:50000]
        self.data.append(data_source)
        self.labels.append(label_source)

        if mode == "train":
          for i in range(0,n_val_per_class):
            tmp_idx = np.where(label_source == i)[0]
            np.random.shuffle(tmp_idx)
            #tmp_idx = torch.from_numpy(tmp_idx)
            img_val = data_source[tmp_idx[:10]]
            for idx in range(img_val.size(0)):
              img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
              img_tmp = self.transform(img_tmp)
              self.data_val.append(img_tmp.unsqueeze(0))
              class_labels_val = label_source[tmp_idx[idx]]
              self.labels_val.append(class_labels_val.unsqueeze(0).float())

          #print(self.data_val[:3])
          self.data_val, self.labels_val = shuffle(self.data_val, self.labels_val, random_state=0)
          self.data_val = torch.cat(self.data_val, dim=0)
          self.labels_val = torch.cat(self.labels_val)
          #BACKGROUNDFLIP: 40% noise ratio => 40% of the images flip to class 0
          l = int(noise_ratio*data_source.shape[0])
          #label_source[:l] = 0
          for i in range(l):
            other_labels = list(range(0, label_source[i])) + list(range(label_source[i+1], 10))
            label_source[i] = random.choice(other_labels)
          self.labels.append(label_source)

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

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
      
class MNIST_BKGNDFLIP():
  def __init__(self, noise_ratio=0.4, n_val_per_class=10, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('data',train=True, download=True)
        else:
            self.mnist = datasets.MNIST('data',train=False, download=True)
            n_val = 0
        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        
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
        data_source = data_source[:50000]
        label_source = label_source[:50000]
        self.data.append(data_source)
        self.labels.append(label_source)

        if mode == "train":
          for i in range(0,n_val_per_class):
            tmp_idx = np.where(label_source == i)[0]
            np.random.shuffle(tmp_idx)
            #tmp_idx = torch.from_numpy(tmp_idx)
            img_val = data_source[tmp_idx[:10]]
            for idx in range(img_val.size(0)):
              img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
              img_tmp = self.transform(img_tmp)
              self.data_val.append(img_tmp.unsqueeze(0))
              class_labels_val = label_source[tmp_idx[idx]]
              self.labels_val.append(class_labels_val.unsqueeze(0).float())

          #print(self.data_val[:3])
          self.data_val, self.labels_val = shuffle(self.data_val, self.labels_val, random_state=0)
          self.data_val = torch.cat(self.data_val, dim=0)
          self.labels_val = torch.cat(self.labels_val)
          #BACKGROUNDFLIP: 40% noise ratio => 40% of the images flip to class 0
          l = int(noise_ratio*data_source.shape[0])
          label_source[:l] = 0
          self.labels.append(label_source)

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)


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

def get_mnist_bkgnd_flip_loader(batch_size, noise_ratio=0.4, n_val_per_class=10, mode='train'):
    """Build and return data loader."""

    dataset = MNIST_BKGNDFLIP(noise_ratio=noise_ratio, n_val_per_class=n_val_per_class, mode=mode )

    shuffle = False
    if mode == 'train':
        shuffle = True
    shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle, num_workers=2)
    return data_loader

def get_mnist_unif_flip_loader(batch_size, noise_ratio=0.4, n_val_per_class=10, mode='train'):
    """Build and return data loader."""

    dataset = MNIST_UNIFFLIP(noise_ratio=noise_ratio, n_val_per_class=n_val_per_class, mode=mode )

    shuffle = False
    if mode == 'train':
        shuffle = True
    shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle, num_workers=2)
    return data_loader