import torch
import torch.nn as nn
import torch.nn.functional as F

class OETrainer():
    def __init__(self, **kwargs):
        self.train_loader = kwargs['train_loader']
        self.valid_loader = kwargs['valid_loader']
        self.out_loader = kwargs['out_loader']

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
        for batch_idx, (data1, data2) in enumerate(zip(self.train_loader, self.out_loader)):
            inputs1, targets = data1
            inputs2, _ = data2
            
            inputs = torch.cat([inputs1, inputs2], dim=1)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs[:len(targets)], targets)
            loss += 0.5 * -(outputs[len(targets):].mean(1) - torch.logsumexp(outputs[len(targets):], dim=1)).mean()
            
            loss.backward()             

            self.optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
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