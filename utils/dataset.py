import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# In ResNet paper, BATCH_SIZE is set as 128
# but in Rethinking2019 paper, a different training strategy is used, BATCH_SIZE is 64
# to completely reproduce, we use new training strategy
def cifar10_dataset(BATCH_SIZE=64):
    # CIFAR 10 statics
    print("INFO: Loading CIFAR10 training dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
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