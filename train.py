import os
import torch
import torchvision
import argparse
import timm
import numpy as np

import utils
import prehoc

# torch.manual_seed(0)

trainers = {
    'ce': prehoc.CETrainer,
    'focal': prehoc.FocalTrainer,
    'oe': prehoc.OETrainer,
    'oecc': prehoc.OECCTrainer,
    'mixup': prehoc.MixupTrainer
}


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)

    parser.add_argument('--inlier-data', '-i', type=str)
    parser.add_argument('--outlier-data', '-o', type=str)
    parser.add_argument('--method', '-m', type=str, choices=['ce', 'focal', 'soft', 'mixup', 'oe', 'oecc'])

    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.inlier_data+'.json')

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    device = 'cuda:0'#+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    wd = 5e-04
    lrde = [50, 75, 90]
        
    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    with open('{}/{}'.format(save_path, 'args.txt'), 'w') as f:
        for a in vars(args):
            f.write('{}: {}\n'.format(a, vars(args)[a]))

    if 'cifar' in args.inlier_data:
        train_loader, valid_loader = utils.get_cifar(args.inlier_data, dataset_path, batch_size)
    elif 'svhn' in args.inlier_data:
        train_loader, valid_loader = utils.get_train_svhn(dataset_path, batch_size)
    elif 'ham10000' in args.inlier_data:
        train_loader, valid_loader = utils.get_imagenet('ham10000', dataset_path, batch_size)
    else:
        train_loader, valid_loader = utils.get_imagenet(args.inlier_data, dataset_path, batch_size)

    if 'resnet18' in args.net:
        # model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        model = utils.ResNet18(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
    elif 'wrn28' in args.net:
        model = utils.WideResNet(28, num_classes, widen_factor=10)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
    elif 'wrn40' in args.net:
        model = utils.WideResNet(40, num_classes, widen_factor=2)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
    elif 'vgg11' == args.net:       
        model = utils.VGG('VGG11', num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
        max_epoch = 200
        lrde = [100, 150, 180]
    elif 'resnet34' in args.net:
        model = utils.ResNet34(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)

    elif 'resnet50' in args.net:
        model = utils.ResNet50(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
    else:
        model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
        wd = 1e-04
        lrde = [30, 60, 90]
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)    

    if not args.outlier_data is None:
        out_loader = utils.get_out(args.outlier_data)
    else:
        out_loader = None 

    if args.method == 'mixup':
        out_loader, _ = utils.get_cifar(args.inlier_data, dataset_path, batch_size)
    A_tr = { # for OECC
        'cifar10':0.9432,
        'cifar100':0.7644,
        'svhn': 0.9500,
        'aircraft':0.76,
        'cub':0.8,
        'scars':0.8
    }
    A_tr = A_tr[args.inlier_data]


    trainer = trainers[args.method](
        model = model,
        train_loader = train_loader, 
        valid_loader = valid_loader,
        out_loader = out_loader,
        
        optimizer = optimizer,
        scheduler = scheduler,
        saver = saver,
        device = device,

        num_classes = num_classes,
        A_tr = A_tr
        )
    
    for epoch in range(max_epoch):
        ## training
        trainer.train()

        ## validation
        trainer.validation(epoch)

if __name__ =='__main__':
    train()