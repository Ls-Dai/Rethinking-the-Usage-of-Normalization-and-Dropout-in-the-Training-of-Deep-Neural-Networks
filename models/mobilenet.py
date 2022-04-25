import torch
import torch.nn as nn
import torch.nn.functional as F

# MobileNetV1 Architecture
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    
    def __init__(self, cfg, num_classes):
        super(MobileNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# MobileNetV1 with IC Layer
class IC_layer(nn.Module):
    def __init__(self, planes, p=0.05, dim ='2d'):
        super(IC_layer, self).__init__()
        if dim == '2d':
            self.bn = nn.BatchNorm2d(planes)
        elif dim == '1d':
            self.bn = nn.BatchNorm1d(planes)
        
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, x):
        return self.dropout(self.bn(x))
    
class Block_IC(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block_IC, self).__init__()
        self.ic1 = IC_layer(in_planes)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.ic2 = IC_layer(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        out = F.relu(self.conv1(self.ic1(x)))
        out = F.relu(self.conv2(self.ic2(out)))
        return out
    
class MobileNet_IC(nn.Module):
    
    def __init__(self, cfg, num_classes):
        super(MobileNet_IC, self).__init__()
        self.cfg = cfg
        self.ic1 = IC_layer(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = self._make_layers(in_planes=32)
        self.ic2 = IC_layer(1024, dim='1d')
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block_IC(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(self.ic1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(self.ic2(out))
        return out

# model functions

def mobilenet(num_classes):
    print("INFO: Creating Mobilenet Model")
    
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model = MobileNet(cfg, num_classes)
    model.model_name = 'mobilenet'
    
    return model

def mobilenet_ic(num_classes):
    print("INFO: Creating Mobilenet Model with IC layer")
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model = MobileNet_IC(cfg, num_classes)
    model.model_name = 'mobilenet_ic'
    return model