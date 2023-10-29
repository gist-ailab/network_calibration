#
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer = None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        if norm_layer is not None:
            self.norm_layer = norm_layer
            self.forward = self.forward_norm

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def set_gamma(self, train_loader, device, norm_layer):
        lambda_ratios=[]
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                out, features = self.forward_features_norm(inputs)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                last_norm = torch.norm(out, p=2, dim=1)

                for i in range(len(features)):
                    x = features[i]
                    norm_clean = torch.norm(x, p=2, dim=[2, 3]).mean(1)
                    
                    lambda_ratio = (last_norm/norm_clean).mean()
                    lambda_ratios.append(lambda_ratio)

        lambda_ratios = torch.tensor(lambda_ratios).view(-1, len(features))
        self.gamma = lambda_ratios.mean(0)[norm_layer]
        print(self.gamma)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

    def forward_norm(self, x):
        out, features = self.forward_features_norm(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = out / torch.norm(out, p=2, dim=1, keepdim=True) * torch.norm(F.relu(features[self.norm_layer]), p=2, dim=[1, 2, 3]).view(-1, 1)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True) * torch.norm(F.relu(features[self.norm_layer]), p=2, dim=[2, 3]).mean(1, keepdim=True) * self.gamma
        out = self.fc(out)
        
        return out

    def forward_features_norm(self, x):
        features = []
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        features.append(out)

        for layer in self.layer1:
            out = layer(out)
            features.append(out)

        for layer in self.layer2:
            out = layer(out)
            features.append(out)

        for layer in self.layer3:
            out = layer(out)
            features.append(out)

        for layer in self.layer4:
            out = layer(out)
            features.append(out)

        return out, features


def ResNet18(num_classes, norm_layer=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, norm_layer=norm_layer)

def ResNet34(num_classes, norm_layer=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes, norm_layer=norm_layer)

def ResNet50(num_classes, norm_layer=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes, norm_layer=norm_layer)
