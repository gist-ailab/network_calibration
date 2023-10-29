#
import torch
import torch.nn as nn
import torch.nn.functional as F


class CETrainer():
    def __init__(self, **kwargs):
        self.train_loader = kwargs['train_loader']
        self.valid_loader = kwargs['valid_loader']

        self.model =  kwargs['model']
        self.optimizer =  kwargs['optimizer']
        self.saver =  kwargs['saver']
        self.scheduler =  kwargs['scheduler']
        self.device =  kwargs['device']

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()            
            self.optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs.max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')    

        self.train_accuracy = correct/total
        self.train_avg_loss = total_loss/len(self.train_loader)
        print()

    def validation(self, epoch):
        self.model.eval()
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()           
        valid_accuracy = correct/total
        self.scheduler.step()
        self.saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, self.train_avg_loss, self.train_accuracy, valid_accuracy))
        print(self.scheduler.get_last_lr())
