import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# ResNet Model Architecture Definition
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper uses option A.
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.model_name = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet witt IC layer
class IC_layer(nn.Module):
    def __init__(self, planes, p=0.5):
        super(IC_layer, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, x):
        return self.dropout(self.bn(x))
    
class BasicBlock_IC(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_IC, self).__init__()
        self.ic1 = IC_layer(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ic2 = IC_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper uses option A.
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(self.ic1(F.relu(x)))
        out = self.conv2(self.ic2(F.relu(out)))
        out += self.shortcut(x)
        return out

class ResNet_IC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_IC, self).__init__()
        self.in_planes = 16
        
        self.ic1 = IC_layer(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.ic2 = IC_layer(64)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.model_name = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(self.ic1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.ic2(out) # do we need this? If yes, put it before or after average pooling?
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out

# model functions
def resnet110(num_classes):
    print("INFO: Creating resnet110 model")
    model = ResNet(BasicBlock, [18,18,18], num_classes=num_classes)
    model.model_name = 'resnet110'
    return model

def resnet164(num_classes):
    print("INFO: Creating resnet164 model")
    model = ResNet(BasicBlock, [27,27,27], num_classes=num_classes)
    model.model_name = 'resnet164'
    return model

def resnet110_ic(num_classes):
    print("INFO: Creating resnet110 model with IC layer")
    model = ResNet_IC(BasicBlock_IC, [18,18,18], num_classes=num_classes)
    model.model_name = 'resnet110_ic'
    return model

def resnet164_ic(num_classes):
    print("INFO: Creating resnet164 model with IC layer")
    model = ResNet_IC(BasicBlock_IC, [27,27,27], num_classes=num_classes)
    model.model_name = 'resnet164_ic'
    return model
