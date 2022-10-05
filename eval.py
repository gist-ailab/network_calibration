import os
import torch
import torchvision
import argparse
import timm
import numpy as np

import utils
import prehoc
import posthoc

# torch.manual_seed(0)

evaluaters = {
    'baseline': posthoc.BaselineEvaluater,
    'temperature': posthoc.TemperatureEvaluater,
    'vector': posthoc.VectorScalingEvaluater,
    'matrix': posthoc.MatrixScalingEvaluater,
    'spline': posthoc.SplineEvaluater,
    'norm': posthoc.NormEvaluater

}


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)

    parser.add_argument('--inlier-data', '-i', type=str)
    parser.add_argument('--method', '-m', type=str, choices=['baseline', 'temperature', 'vector', 'matrix', 'spline', 'norm'])

    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.inlier_data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    if args.net == 'resnet18':
        batch_size = int(config['batch_size'])
        
    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    with open('{}/{}'.format(save_path, 'args.txt'), 'w') as f:
        for a in vars(args):
            f.write('{}: {}\n'.format(a, vars(args)[a]))

    if 'cifar' in args.inlier_data:
        train_loader, valid_loader = utils.get_cifar(args.inlier_data, dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.1)


    if 'resnet18' in args.net:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    

    evaluater = evaluaters[args.method](
        model = model,
        train_loader = train_loader, 
        valid_loader = valid_loader,
        test_loader = test_loader,
        
        device = device,
        num_classes = num_classes,
        ndim = 512
        )
    
    ## evaluation
    evaluater.eval()


if __name__ =='__main__':
    train()