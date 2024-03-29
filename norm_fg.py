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


def forward_norm(model, loader, device):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    original_features_norm = []
    corrupted_features_norm = []

    model.eval()

    ratios = []
    lambda_ratios = []

    features_norm = []
    upsampler = torch.nn.Upsample(size=[448,448])
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

            last_feat_clean, clean_features = model.forward_features_norm(inputs)
            _, corrupted_features = model.forward_features_norm(corrputed_inputs)
            last_feat_clean = F.avg_pool2d(last_feat_clean, 4)
            last_feat_clean = last_feat_clean.view(last_feat_clean.size(0), -1)

            last_norm = torch.norm(last_feat_clean, p=2, dim=1)

            for i in range(len(clean_features)):
                # norm_clean = torch.norm(clean_features[i], p=2, dim=[2,3])
                # norm_corrupt = torch.norm(corrupted_features[i], p=2, dim=[2,3])
                x = clean_features[i]
                norm_clean = torch.norm(x, p=2, dim=[2, 3]).mean(1)#.view(-1, 1)
                # norm_clean = torch.norm(x, dim=[2,3], p=2)
                # norm_clean = torch.where(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]) < 0, -torch.abs(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3])).sqrt(), torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]).sqrt())
                # norm_clean = torch.norm
                x = corrupted_features[i]
                norm_corrupt = torch.norm(x, p=2, dim=[2, 3]).mean(1)#.view(-1, 1)
                # norm_corrupt = torch.norm(x, dim=[2,3], p=2)
                # norm_corrupt = torch.where(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]) < 0, -torch.abs(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3])).sqrt(), torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]).sqrt())
                
                ratio = (norm_clean/norm_corrupt).mean()
                ratios.append(ratio)
                
                lambda_ratio = (last_norm/norm_clean).mean()
                lambda_ratios.append(lambda_ratio)
                # print(norm_clean.mean(), norm_corrupt.mean())

    ratios = torch.tensor(ratios).view(-1, len(clean_features))
    lambda_ratios = torch.tensor(lambda_ratios).view(-1, len(clean_features))

    print(ratios.mean(dim=0), ratios.shape)
    # print(lambda_ratios.mean(dim=0))


def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.fc(out)

    return out


def forward_features(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    return out


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


timm.models.ResNet.forward_features_norm = forward_features_norm


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', 
                        default = 'resnet50', type=str)
    parser.add_argument('--gpu', '-g', 
                        default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--inlier-data', '-i', type=str)
    args = parser.parse_args()

    # 
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

    print(len(valid_loader), len(test_loader))

    if 'resnet50' in args.net:
        model = timm.create_model(args.net, num_classes = num_classes)
        # model = utils.ResNet50(num_classes=num_classes)
        # model = utils.ResNet50(num_classes=num_classes, norm_layer = -2)

        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))

    if 'resnet152' in args.net:
        model = timm.create_model(args.net, num_classes = num_classes)
        # model = utils.ResNet50(num_classes=num_classes)
        # model = utils.ResNet50(num_classes=num_classes, norm_layer = -2)

        model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
        
    model.to(device)

    forward_norm(model, train_loader, device)


if __name__ =='__main__':
    train()
