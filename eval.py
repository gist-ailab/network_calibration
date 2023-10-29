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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'wrn40', type=str)
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

    if 'cifar' in args.inlier_data:
        train_loader, valid_loader = utils.get_cifar(args.inlier_data, dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)
    elif 'svhn' in args.inlier_data:
        train_loader, valid_loader = utils.get_train_svhn(dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)

    elif 'ham10000' in args.inlier_data:
        train_loader, valid_loader = utils.get_imagenet('ham10000', dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)
    else:
        train_loader, valid_loader = utils.get_imagenet(args.inlier_data, dataset_path, batch_size)
        valid_loader, test_loader = utils.devide_val_test(valid_loader, 0.9)
    print(len(valid_loader), len(test_loader))

    if 'resnet18' in args.net:
        # model = utils.ResNet18(num_classes = num_classes)
        def forward_features_norm(self, x):
            features = []
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)       
            x = self.layer4[0](x)
            # norm = torch.norm(x, p=2, dim=[2,3])
            x_ = self.global_pool(x).view(-1, x.size(1))
            norm = torch.norm(x_, p=2, dim=1, keepdim=True)
            x = self.layer4[1](x)

            x = self.global_pool(x).view(-1, x.size(1))
            # print(torch.norm(x, p=2, dim=1, keepdim=True).shape, norm.mean(dim=1, keepdim=True).shape)
            x = x / torch.norm(x, p=2, dim=1, keepdim=True) * norm
            outputs = model.fc(x)
            return outputs

        # timm.models.ResNet.forward = forward_features_norm
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    elif 'wrn28' in args.net:
        def forward_features(self,x):
            out = self.conv1(x)
            # print(out.shape)
            out = self.block1(out)
            out = self.block2(out)

            out = self.block3.layer[0](out)
            out = self.block3.layer[1](out)
            out = self.block3.layer[2](out)
            norm1 = torch.norm(F.relu(out), p=2, dim=[2,3])
            norm1 = norm1.mean(dim=1)
            out = self.block3.layer[3](out)
            norm2 = torch.norm(F.relu(out), p=2, dim=[2,3])
            norm2 = norm2.mean(dim=1)
            out = self.relu(self.bn1(out))

            return out / norm2.view(-1, 1, 1, 1) * norm1.view(-1, 1, 1, 1)
        # utils.WideResNet.forward_features = forward_features

        model = utils.WideResNet(28, num_classes, widen_factor=10)
        # model = utils.WideResNet(28, num_classes, widen_factor=10)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    elif 'vgg11' == args.net:       
        model = utils.VGG('VGG11', num_classes)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    elif 'wrn40' in args.net:
        model = utils.WideResNet(40, num_classes, widen_factor=2)
        # model = utils.WideResNet(40, num_classes, widen_factor=2)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    elif 'resnet50' in args.net:
        model = utils.ResNet50(num_classes=num_classes)
        # model = utils.ResNet50(num_classes=num_classes, norm_layer = -2)

        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))

    elif 'resnet34' in args.net:
        # model = utils.ResNet34(num_classes=num_classes)
        model = utils.ResNet34(num_classes=num_classes, norm_layer = -1)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))

    else:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        if args.inlier_data == 'imagenet':
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)       
                x = self.layer2(x)
                x = self.layer3(x)       

                x = self.layer4[0](x)      
                x = self.layer4[1](x)      
                norm1 = torch.sum(x**2, dim=[2,3]).sqrt()
                norm1 = norm1.mean(dim=1)
                x = self.layer4[2](x)
                norm2 = torch.sum(x**2, dim=[2,3]).sqrt()
                norm2 = norm2.mean(dim=1)
                x = x/norm2.view(-1,1,1,1) * norm1.view(-1,1,1,1)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)

                x = self.fc(x)
                return x
            # torchvision.models.ResNet._forward_impl = forward            
            model = torchvision.models.resnet50(pretrained=True)
        else:
            model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()
    # model.set_gamma(train_loader, device, -1)
    
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
