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

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
# test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


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

    if not hasattr(dataset, 'targets'):
        targets = []
        for data in dataset:
            targets.append(data[1])
    else:
        targets = dataset.targets
    indices = list(range(len(dataset)))
    val_indices, test_indices = train_test_split(indices, test_size=divide_rate, stratify=targets) 

    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, num_workers=2, batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=100)
    return val_loader, test_loader


def get_train_svhn(folder, batch_size):
    transform = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])

    train_data = dset.SVHN(folder, split='train', transform=transform, download=True)    
    test_data = dset.SVHN(folder, split='test', transform=transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)     
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


def get_aircraft(imagenet_path, batch_size=32, jigsaw=False, eval=False):
    train_transforms = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225]) 
    ])
    test_transforms = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225]),
    ])

    train_trans = train_transforms
    test_trans = test_transforms
    if eval:
        train_trans = test_transforms    
    trainset = torchvision.datasets.FGVCAircraft(imagenet_path, split='trainval', transform= train_trans, download=True)
    testset = torchvision.datasets.FGVCAircraft(imagenet_path, split='test', transform= test_trans, download=True)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_scars(imagenet_path, batch_size=32, eval=False):
    train_transforms = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225]) 
    ])
    test_transforms = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225]),
    ])

    train_trans = train_transforms
    test_trans = test_transforms
    if eval:
        train_trans = test_transforms

    trainset = torchvision.datasets.StanfordCars(imagenet_path, split='train', transform= train_trans, download=True)
    testset = torchvision.datasets.StanfordCars(imagenet_path, split='test', transform= test_trans, download=True)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_food101(imagenet_path, batch_size=32, jigsaw = False, eval=False):
    train_transforms = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225]) 
    ])
    test_transforms = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225]),
    ])

    train_trans = train_transforms
    test_trans = test_transforms
    if eval:
        train_trans = test_transforms

    trainset = torchvision.datasets.Food101(imagenet_path, split='train', transform= train_trans, download=True)
    testset = torchvision.datasets.Food101(imagenet_path, split='test', transform= test_trans, download=True)
    if jigsaw:
        trainset = jigsaw_train_dataset(trainset, train_trans)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader
    
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

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


class cifar10Nosiy(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0, asym=False):
        super(cifar10Nosiy, self).__init__(root, transform=transform, target_transform=target_transform)
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return
        
if __name__ == '__main__':
    pass