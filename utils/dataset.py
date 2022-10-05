import json
import torch.utils.data as data
import numpy as np
import torch
import os
import random

from PIL import Image
from scipy import io
from torchvision import transforms
from torchvision import datasets as dset
from skimage.filters import gaussian as gblur
from sklearn.model_selection import train_test_split
import torchvision


train_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor()])
# test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])


train_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]) 
])

test_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]) 
])
def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_cifar(dataset, folder, batch_size):
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 10
    else:
        train_data = dset.CIFAR100(folder, train=True, transform=train_transform_cifar, download=True)
        test_data = dset.CIFAR100(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

def devide_val_test(valid_loader, divide_rate = 0.1, seed = 0):
    np.random.seed(0)
    dataset = valid_loader.dataset

    indices = list(range(len(dataset)))
    val_indices, test_indices = train_test_split(indices, test_size=divide_rate, stratify=dataset.targets) 

    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, num_workers=2, batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=100)
    return val_loader, test_loader


def get_train_svhn(folder, batch_size):
    train_data = dset.SVHN(folder, split='train', transform=test_transform_cifar, download=True)    
    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)     
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return train_loader, valid_loader
    
def get_outlier(path, batch_size):
    class temp(torch.utils.data.Dataset):
        def __init__(self, path, transform=None):
            self.data = np.load(path)
            self.transform = transform

        def __getitem__(self, index):
            data = self.data[index]
            data = self.transform(data)
            return data

        def __len__(self):
            return len(self.data)
    
    test_data = temp(path, transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)    
    return valid_loader

def get_tinyimagenet(path, batch_size):
    class TinyImages(torch.utils.data.Dataset):
        def __init__(self, path, transform=None, exclude_cifar=True):
            data_file = open(path+'/', "rb")

            def load_image(idx):
                data_file.seek(idx * 3072)
                data = data_file.read(3072)
                return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

            self.load_image = load_image
            self.offset = 0     # offset index

            self.transform = transform
            self.exclude_cifar = exclude_cifar

            if exclude_cifar:
                self.cifar_idxs = []
                with open(path+'/80mn_cifar_idxs.txt', 'r') as idxs:
                    for idx in idxs:
                        # indices in file take the 80mn database to start at 1, hence "- 1"
                        self.cifar_idxs.append(int(idx) - 1)

                # hash table option
                self.cifar_idxs = set(self.cifar_idxs)
                self.in_cifar = lambda x: x in self.cifar_idxs

        def __getitem__(self, index):
            index = (index + self.offset) % 79302016

            if self.exclude_cifar:
                while self.in_cifar(index):
                    index = np.random.randint(79302017)

            img = self.load_image(index)
            if self.transform is not None:
                img = self.transform(img)

            return img, 0  # 0 is the class
        def __len__(self):
            return 79302017

    ood_data = TinyImages(path, test_transform_cifar, True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return ood_loader

def get_svhn(folder, batch_size):
    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader

def get_textures(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader

def get_lsun(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader    

def get_places(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_data.samples = random.sample(ood_data.samples, 10000)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_folder(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_blob():
    # /////////////// Blob ///////////////
    ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, 32, 32, 3)))
    for i in range(10000):
        ood_data[i] = gblur(ood_data[i], sigma=1.5, channel_axis=False)
        ood_data[i][ood_data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True, pin_memory=True)
    return ood_loader

def get_gaussian():
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(10000, 3, 32, 32), scale=0.5), -1, 1))
    )
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size = 100, shuffle=True)
    return ood_loader

def get_rademacher():
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(10000, 3, 32, 32)).astype(np.float32)) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True)
    return ood_loader

def get_imagenet(dataset, imagenet_path, batch_size=32):
    trainset = torchvision.datasets.ImageFolder(imagenet_path+'/train', train_transforms)
    testset = torchvision.datasets.ImageFolder(imagenet_path+'/val', test_transforms)

    if dataset == 'imagenet100':
        new_samples = []
        for sample in trainset.samples:
            if sample[1] in list(range(0, 100)):
                new_samples.append(sample)
        trainset.samples = new_samples

        new_samples = []
        for sample in testset.samples:
            if sample[1] in list(range(0, 100)):
                new_samples.append(sample)
        testset.samples = new_samples

    if dataset == 'imagenet10':
        new_samples = []
        for sample in trainset.samples:
            if sample[1] in list(range(0, 10)):
                new_samples.append(sample)
        trainset.samples = new_samples

        new_samples = []
        for sample in testset.samples:
            if sample[1] in list(range(0, 10)):
                new_samples.append(sample)
        testset.samples = new_samples
        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_ood_folder(path, batch_size = 32):
    oodset = torchvision.datasets.ImageFolder(path, test_transforms)
    ood_loader = torch.utils.data.DataLoader(oodset, batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    return ood_loader
    
if __name__ == '__main__':
    pass