import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def cost_CONF_mixup_interpolated(o,t1,t2,lam):# \loss is applied over the mixup Image. 
    COST=None
    acc1=(o.argmax(dim=1)==t1).sum()#/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
    acc2=(o.argmax(dim=1)==t2).sum()#/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
    acc=(lam*acc1+(1-lam)*acc2)/float(t1.size(0))

    predicted_prob=nn.functional.softmax(o,dim=1)
    predicted_prob,index = torch.max(predicted_prob,1)

    calib_cost_index = 1
    if calib_cost_index==1:
        COST=((predicted_prob.mean()-acc)**2)	
    elif calib_cost_index==2:
        COST=((predicted_prob-acc)**2).mean()

    lamda = 4
    return lamda*COST

class MixupTrainer():
    def __init__(self, **kwargs):
        self.train_loader = kwargs['train_loader']
        self.valid_loader = kwargs['valid_loader']
        self.train_loader2 = kwargs['out_loader']

        self.model =  kwargs['model']
        self.optimizer =  kwargs['optimizer']
        self.saver =  kwargs['saver']
        self.scheduler =  kwargs['scheduler']
        self.device =  kwargs['device']

        self.criterion = nn.CrossEntropyLoss()
        self.mixup_coeff = 1

    def train(self):
        self.model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (data1, data2) in enumerate(zip(self.train_loader, self.train_loader2)):
            inputs1, targets1 = data1
            inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)

            inputs2, targets2 = data2
            inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

            lam = np.random.beta(self.mixup_coeff, self.mixup_coeff)
            inputs = lam * inputs1 + (1-lam) * inputs2

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = lam * self.criterion(outputs, targets1) + (1-lam) * self.criterion(outputs, targets2)

            # calibration loss  
            loss += cost_CONF_mixup_interpolated(outputs,targets1,targets2,lam) 


            loss.backward()        
                
            self.optimizer.step()

            total_loss += loss
            total += targets1.size(0)
            _, predicted = outputs.max(1)            
            correct += predicted.eq(targets1).sum().item()            
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