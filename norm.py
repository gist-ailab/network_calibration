import os
import torch
import torch.nn.functional as F
import torchvision
import argparse
import timm
import numpy as np

import utils
import prehoc
import posthoc


def forward_norm(model, loader, device):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    original_features_norm = []
    corrupted_features_norm = []
    model.eval()

    ratios = []

    upsampler = torch.nn.Upsample(size=[32,32])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            features = features.mean(dim=1)
            # obtain attention map
            attn_map = torch.ones_like(features.view(-1, features.size(1)*features.size(2)))
            attn_idx = features.view(-1, features.size(1)*features.size(2)).max(dim=1).indices
            # need to changed later
            for i in range(len(attn_idx)):
                attn_map[i][attn_idx[i]] = 0.0
            # change the size of attention map as input size
            attn_map = attn_map.view(-1, 1, features.size(1), features.size(2))
            upsampled_attn_map = upsampler(attn_map)
            # print(upsampled_attn_map.cpu().numpy())

            corrputed_inputs = inputs * upsampled_attn_map

            _, clean_features = model.forward_features_norm(inputs)
            _, corrupted_features = model.forward_features_norm(corrputed_inputs)

            for i in range(len(clean_features)):
                # norm_clean = torch.norm(clean_features[i], p=2, dim=[2,3])
                # norm_corrupt = torch.norm(corrupted_features[i], p=2, dim=[2,3])
                x = clean_features[i]
                norm_clean = torch.where(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]) < 0, -torch.abs(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3])).sqrt(), torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]).sqrt())
                x = corrupted_features[i]
                norm_corrupt = torch.where(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]) < 0, -torch.abs(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3])).sqrt(), torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]).sqrt())
                ratio = (norm_clean/norm_corrupt).mean()
                ratios.append(ratio)

    ratios = torch.tensor(ratios).view(-1, len(clean_features))
    print(ratios.mean(dim=0), ratios.shape)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--inlier-data', '-i', type=str)
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
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    elif 'wrn28' in args.net:
        model = utils.WideResNet(28, num_classes, widen_factor=10)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    elif 'vgg11' == args.net:  
        model = utils.VGG('VGG11', num_classes)
        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    else:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        if args.inlier_data == 'imagenet':   
            model = torchvision.models.resnet50(pretrained=True)
        else:
            model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)

    forward_norm(model, train_loader, device)


if __name__ =='__main__':
    train()