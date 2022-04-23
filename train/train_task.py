import torch
import torch.nn as nn
from models.resnet import resnet110, resnet110_ic
from models.densenet import densenet40, densenet40_ic
from models.vgg import vgg16_bn, vgg16_bn_ic
from train.training import Learner

__all__ = ['resnet110', 'resnet110_ic', 'vgg16_bn', 'vgg16_bn_ic', 'densenet40', 'densenet40_ic']

def train_CIFAR10(model_type, train_config=None):
    
    # define model instance
    if model_type in __all__:
        model = globals()[model_type](num_classes=10)
    else:
        raise Exception(f"Current model not implemented, avaliable models are:{__all__}")
    
    # Training config for CIFAR10
    if train_config is None:
        print("INFO: Using default training config")
        lr = 0.001
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss().cuda()
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)

        train_config = {'model':model,
                        'loss_fn':loss_fn,
                        'optim':optimizer,
                        'scheduler':lr_scheduler,
                        'epochs':epochs}
    else:
        print("INFO: Using self-defined training config")

    # train
    learner = Learner(task='cifar10', train_config=train_config)
    res = learner.train()
    
    return res

def train_CIFAR100(model_type, train_config=None):
    
    # define model instance
    if model_type in __all__:
        model = globals()[model_type](num_classes=100)
    else:
        raise Exception(f"Current model not implemented, avaliable models are:{__all__}")
    
    # Training config for CIFAR100
    if train_config is None:
        print("INFO: Using default training config")
        lr = 0.001
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss().cuda()
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)

        train_config = {'model':model,
                        'loss_fn':loss_fn,
                        'optim':optimizer,
                        'scheduler':lr_scheduler,
                        'epochs':epochs}
    else:
        print("INFO: Using self-defined training config")
    
    # train
    learner = Learner(task='cifar100', train_config=train_config)
    res = learner.train()
    
    return res