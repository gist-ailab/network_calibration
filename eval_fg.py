#
import os
import argparse

import torch
import torch.nn.functional as F
import torchvision

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
    'norm': posthoc.NormEvaluater,
    'vectornorm': posthoc.VectorNormEvaluater,
    'normonly': posthoc.OnlyNormEvaluater

}


def forward_features_norm(self, x):
    features = []
    out = self.conv1(x)
    out = F.relu(self.bn1(out))
    # features.append(out)
    out = self.maxpool(out)
    for layer in self.layer1:
        out = layer(out)
        # features.append(out)

    for layer in self.layer2:
        out = layer(out)
        # features.append(out)

    for layer in self.layer3:
        out = layer(out)
        features.append(out)

    for layer in self.layer4:
        out = layer(out)
        features.append(out)

    return out, features


def set_gamma(self, train_loader, device, norm_layer):
    lambda_ratios=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            out, features = self.forward_features_norm(inputs)
            out = F.avg_pool2d(out, 14)
            out = out.view(out.size(0), -1)
            last_norm = torch.norm(out, p=2, dim=1)

            for i in range(len(features)):
                x = features[i]
                norm_clean = torch.norm(F.relu(x), p=2, dim=[2, 3]).mean(1)
                
                lambda_ratio = (last_norm/norm_clean).mean()
                lambda_ratios.append(lambda_ratio)

    lambda_ratios = torch.tensor(lambda_ratios).view(-1, len(features))
    self.gamma = lambda_ratios.mean(0)[norm_layer]
    print(self.gamma)


def forward_norm(self, x):
    out, features = self.forward_features_norm(x)
    out = F.avg_pool2d(out, 14)
    out = out.view(out.size(0), -1)
    # out = out / torch.norm(out, p=2, dim=1, keepdim=True) * torch.norm(F.relu(features[self.norm_layer]), p=2, dim=[1, 2, 3]).view(-1, 1)
    out = out / torch.norm(out, p=2, dim=1, keepdim=True) * torch.norm(F.relu(features[self.norm_layer]), p=2, dim=[2, 3]).mean(1, keepdim=True) * self.gamma
    out = self.fc(out)
    return out


timm.models.ResNet.forward_features_norm = forward_features_norm
timm.models.ResNet.set_gamma = set_gamma
timm.models.ResNet.forward = forward_norm
timm.models.ResNet.norm_layer = -6


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)

    parser.add_argument('--inlier-data', '-i', type=str)
    parser.add_argument('--method', '-m', type=str, choices=['baseline', 'temperature', 'vector', 'matrix', 'spline', 'norm', 'vectornorm', 'normonly'])

    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.inlier_data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))
    batch_size = int(config['batch_size'])
        
    print(model_name, dataset_path.split('/')[-2], batch_size)
    
    with open('{}/{}'.format(save_path, 'args.txt'), 'w') as f:
        for a in vars(args):
            f.write('{}: {}\n'.format(a, vars(args)[a]))

    if 'aircraft' in args.inlier_data:
        train_loader, valid_loader = utils.get_aircraft(dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)
    elif 'scars' in args.inlier_data:
        train_loader, valid_loader = utils.get_scars(dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)
    elif 'food101' in args.inlier_data:
        train_loader, valid_loader = utils.get_food101(dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)

    print(len(valid_loader), len(test_loader), num_classes)

    if 'resnet50' in args.net:
        model = timm.create_model(args.net, num_classes = num_classes)

        # model = utils.ResNet50(num_classes=num_classes, norm_layer = -2)

        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))

    elif 'resnet34' in args.net:
        # model = utils.ResNet34(num_classes=num_classes)
        # model = utils.ResNet34(num_classes=num_classes, norm_layer = -2)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))

    if 'resnet152' in args.net:
        model = timm.create_model(args.net, num_classes = num_classes)

        # model = utils.ResNet50(num_classes=num_classes, norm_layer = -2)

        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()
    model.set_gamma(train_loader, device, -2)
    
    evaluater = evaluaters[args.method](
        model = model,
        train_loader = train_loader, 
        valid_loader = valid_loader,
        test_loader = test_loader,
        
        device = device,
        num_classes = num_classes,
        ndim = 2048
        )
    
    ## evaluation
    evaluater.eval()


if __name__ =='__main__':
    train()
