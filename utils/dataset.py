import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# In ResNet paper, BATCH_SIZE is set as 128
# but in Rethinking2019 paper, a different training strategy is used, BATCH_SIZE is 64
# to completely reproduce, we use new training strategy
def cifar10_dataset(BATCH_SIZE=64):
    print("INFO: Loading CIFAR10 training dataset")
    # CIFAR 10 statics
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    
    # training set
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32,4),
                                          transforms.ToTensor(),
                                          normalize])
    train_dataset = datasets.CIFAR10(root='./data', 
                                     train=True, 
                                     transform=train_transform, 
                                     download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=BATCH_SIZE, 
                                                   shuffle=True, 
                                                   num_workers=1,
                                                   pin_memory=True)
    
    # test set
    print("INFO: Loading CIFAR10 test dataset")
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(root='./data', 
                                    train=False, 
                                    transform=test_transform, 
                                    download=True)                                                            
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
                                            
    return train_dataloader, test_dataloader


def cifar100_dataset(BATCH_SIZE=64):
    print("INFO: Loading CIFAR100 training dataset")
    # CIFAR 100 statics
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    
    # training set
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32,4),
                                          transforms.ToTensor(),
                                          normalize])
    train_dataset = datasets.CIFAR100(root='./data', 
                                      train=True, 
                                      transform=train_transform, 
                                      download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=BATCH_SIZE, 
                                                   shuffle=True, 
                                                   num_workers=1,
                                                   pin_memory=True)
    
    # test set
    print("INFO: Loading CIFAR100 test dataset")
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(root='./data', 
                                     train=False, 
                                     transform=test_transform, 
                                     download=True)                                                            
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
                                            
    return train_dataloader, test_dataloader