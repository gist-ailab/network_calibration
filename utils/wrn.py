import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, norm_layer = None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.n = n

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        if norm_layer is not None:
            self.norm_layer = norm_layer
            self.forward = self.forward_norm



    def forward(self, x):
        out = self.forward_features(x)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def forward_norm(self, x):
        out, features = self.forward_features_norm(x)
        norm = torch.norm(out, p=2, dim=[2,3]).mean(dim=1, keepdim=True)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        # print(out.shape, features[self.norm_layer].shape
        # out = out/torch.norm(out, p=2, dim=1, keepdim=True) * torch.norm(F.relu(features[self.norm_layer]), p=2, dim=[1,2,3]).mean(dim=1, keepdim=True)
        # x = features[self.norm_layer]
        # norm = torch.where(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]) < 0, -torch.abs(torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3])).sqrt(), torch.sum(torch.where(x>0, (x*x), -(x*x)), dim=[2,3]).sqrt())
        # out = out/torch.norm(out, p=2, dim=1, keepdim=True) * norm
        out = out/torch.norm(out, p=2, dim=1, keepdim=True) * torch.norm(F.relu(features[self.norm_layer]), p=2, dim=[1, 2, 3]).view(-1, 1)# .mean(dim=1, keepdim=True)

        out = self.fc(out)
        return out

    def forward_features(self,x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def forward_features_norm(self, x):
        features = []
        out = self.conv1(x)
        features.append(out)

        for i in range(self.n):
            out_ = out
            block = self.block1.layer[i]
            if not block.equalInOut:
                out_ = block.relu1(block.bn1(out_))
            else:
                out = block.relu1(block.bn1(out_))
            if block.equalInOut:
                out = block.conv1(out)
                features.append(out)
                out = block.relu2(block.bn2(out))
            else:
                out = block.conv1(out_)
                features.append(out)
                out = block.relu2(block.bn2(out))
            if block.droprate > 0:
                out = F.dropout(out, p=block.droprate, training=block.training)
            out = block.conv2(out)
            features.append(out)
            if not block.equalInOut:
                out =  torch.add(block.convShortcut(out_), out)
            else:
                out =  torch.add(out_, out)

        for i in range(self.n):
            out_ = out
            block = self.block2.layer[i]
            if not block.equalInOut:
                out_ = block.relu1(block.bn1(out_))
            else:
                out = block.relu1(block.bn1(out_))
            if block.equalInOut:
                out = block.conv1(out)
                features.append(out)
                out = block.relu2(block.bn2(out))
            else:
                out = block.conv1(out_)
                features.append(out)
                out = block.relu2(block.bn2(out))
            if block.droprate > 0:
                out = F.dropout(out, p=block.droprate, training=block.training)
            out = block.conv2(out)
            features.append(out)
            if not block.equalInOut:
                out =  torch.add(block.convShortcut(out_), out)
            else:
                out =  torch.add(out_, out)

        for i in range(self.n):
            out_ = out
            block = self.block3.layer[i]
            if not block.equalInOut:
                out_ = block.relu1(block.bn1(out_))
            else:
                out = block.relu1(block.bn1(out_))
            if block.equalInOut:
                out = block.conv1(out)
                features.append(out)
                out = block.relu2(block.bn2(out))
            else:
                out = block.conv1(out_)
                features.append(out)
                out = block.relu2(block.bn2(out))
            if block.droprate > 0:
                out = F.dropout(out, p=block.droprate, training=block.training)
            out = block.conv2(out)
            features.append(out)
            if not block.equalInOut:
                out =  torch.add(block.convShortcut(out_), out)
            else:
                out =  torch.add(out_, out)
        out = self.relu(self.bn1(out))
        return out, features