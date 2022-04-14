import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.dataset import cifar10_dataset
from utils.utils import AverageMeter, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner():
    def __init__(self, task, train_config):
        '''
        task: define which task to train
        train_conig: a dict of {'model': model to use,
                                'loss_fn': loss function to use,
                                'optim': optimizer to use,
                                'scheduler': lr scheduler to use,
                                'epochs': how many epochs to train on}
        '''
        # training settings
        if task == 'cifar10':
            self.train_loader, self.test_loader = cifar10_dataset()
        else:
            raise Exception('task not implemented!')
        self.config = train_config
        self.model = train_config['model']
        self.model_name = self.model.model_name
        self.model.to(device)
        self.criterion = train_config['loss_fn']
        self.optimizer = train_config['optim']
        self.scheduler = train_config['scheduler']
        
        # initialization of record variables
        self.test_acc_all = []
        self.best_acc = 0.0
        self.model_path = f'./res/{task}_{self.model_name}_best.pth'
        
    def train(self):
        cudnn.benchmark = True
        epochs = self.config['epochs']
        
        for epoch in tqdm(range(epochs)):
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            
            self.model.train()
            self.train_step(epoch)
            
            self.model.eval()
            test_acc = self.validate(epoch)
            
            if test_acc > self.best_acc:
                self.save_model()
                self.best_acc = test_acc
            
        return self.test_acc_all
    
    def train_step(self, epoch, verbose=True):
        """
        Run one train epoch
        """
        losses = AverageMeter()
        top1_acc = AverageMeter()
        
        for i, (images, labels) in enumerate(self.train_loader):
            
            # fetch batch data
            labels = labels.to(device)
            images = images.to(device)

            # compute output
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            prec1 = accuracy(logits.data, labels)[0]
            losses.update(loss.item(), labels.size(0))
            top1_acc.update(prec1.item(), labels.size(0))

            # measure elapsed time
            if i % 100 == 0 and verbose:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.avg:.4f}\t'
                      'Prec@1 {top1.avg:.3f}'.format(
                          epoch, i, len(self.train_loader), loss=losses, top1=top1_acc))
                
    def validate(self, epoch, verbose=True):
        """
        Run evaluation
        """
        top1_acc = AverageMeter()

        with torch.no_grad():
            for images, labels in self.test_loader:
                
                # fetch batch data
                labels = labels.to(device)
                images = images.to(device)

                # compute output
                logits = self.model(images)

                # measure accuracy and record loss
                prec1 = accuracy(logits.data, labels)[0]
                top1_acc.update(prec1.item(), labels.size(0))
        if verbose:
            print('Epoch[{}] *Validation*: Prec@1 {top1.avg:.3f}'.format(epoch, top1=top1_acc))
        
        self.test_acc_all.append(top1_acc)
        
        return top1_acc.avg
        
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        
    def check_point(self):
        # not implemented yet
        pass
    
    def logging(self):
        # not implemented yet
        pass
