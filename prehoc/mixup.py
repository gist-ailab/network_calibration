import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.functional import softmax

def accuracy_per_bin(predicted,real_tag,n_bins=15,apply_softmax=True):

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1)
	else:
		predicted_prob=predicted
	

	accuracy,index = torch.max(predicted_prob,1)
	selected_label=index.long()==real_tag
	prob=torch.from_numpy(np.linspace(0,1,n_bins+1)).float().cuda()
	acc=torch.from_numpy(np.linspace(0,1,n_bins+1)).float().cuda()
	total_data = len(accuracy)
	samples_per_bin=[]
	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		boolean_upper = accuracy<=max_

		if p==0:#we include the first element in bin
			boolean_down=accuracy>=min_
		else:#after that we included in the previous bin
			boolean_down=accuracy>min_

		index_range=boolean_down & boolean_upper
		label_sel=selected_label[index_range]
		
		if len(label_sel)==0:
			acc[p]=0.0
		else:
			acc[p]=label_sel.sum().float()/float(len(label_sel))

		samples_per_bin.append(len(label_sel))

	samples_per_bin=torch.from_numpy(np.array(samples_per_bin)).cuda()
	acc=acc[0:-1]
	prob=prob[0:-1]
	return acc,prob,samples_per_bin


def average_confidence_per_bin(predicted,n_bins=15,apply_softmax=True):

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1)
	else:
		predicted_prob=predicted
	
	prob=torch.from_numpy(np.linspace(0,1,n_bins+1)).float().cuda()
	conf=torch.from_numpy(np.linspace(0,1,n_bins+1)).float().cuda()
	accuracy,index = torch.max(predicted_prob,1)

	samples_per_bin=[]
	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		
		boolean_upper = accuracy<=max_

		if p==0:#we include the first element in bin
			boolean_down =accuracy>=min_
		else:#after that we included in the previous bin
			boolean_down =accuracy>min_

		index_range=boolean_down & boolean_upper
		prob_sel=accuracy[index_range]
		
		if len(prob_sel)==0:
			conf[p] = 0.0
		else:
			conf[p] = prob_sel.sum().float()/float(len(prob_sel))
            # conf[p]=prob_sel.sum().float()/float(len(prob_sel))
        # samples_per_bin.append(len(prob_sel))
		samples_per_bin.append(len(prob_sel))

	samples_per_bin=torch.from_numpy(np.array(samples_per_bin)).cuda()
	conf=conf[0:-1]
	prob=prob[0:-1]

	return conf,prob,samples_per_bin

def confidence_per_bin(predicted,n_bins=10,apply_softmax=True):

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1)
	else:
		predicted_prob=predicted
	
	prob=torch.from_numpy(np.linspace(0,1,n_bins+1)).float().cuda()
	conf=torch.from_numpy(np.linspace(0,1,n_bins+1)).float().cuda()
	max_confidence,index = torch.max(predicted_prob,1)

	samples_per_bin=[]
	conf_values_per_bin=[]

	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		
		boolean_upper = max_confidence<=max_

		if p==0:#we include the first element in bin
			boolean_down =max_confidence>=min_
		else:#after that we included in the previous bin
			boolean_down =max_confidence>min_

		index_range=boolean_down & boolean_upper
		prob_sel=max_confidence[index_range]
		
		if len(prob_sel)==0:
			conf_values_per_bin.append([0.0])
		else:
			conf_values_per_bin.append(prob_sel)

		samples_per_bin.append(len(prob_sel))

	samples_per_bin=torch.from_numpy(np.array(samples_per_bin)).cuda()
	conf=conf[0:-1]
	prob=prob[0:-1]

	return conf_values_per_bin,prob,samples_per_bin

def calib_cost(a,c,c_per_bin,s,index):
		if index==1:
			return ((c-a.detach())**2)*s
		elif index==2:
			return (((c_per_bin-a.detach())**2).mean())*s	

def cost_CONF_mixup_interpolated(o,t1,t2,lam):
    bins_for_train = [5, 15, 30]
    COST=torch.tensor(0.0).cuda()
    for n_bins in bins_for_train:
        aux_cost,tot_samples=[torch.tensor(0.0).cuda()]*2
        acc1, prob, samples_per_bin1=accuracy_per_bin(o,t1,n_bins)
        acc2, prob, samples_per_bin2=accuracy_per_bin(o,t2,n_bins)
        acc=acc1.clone().detach()

        for idx,(a1,a2,s1,s2) in enumerate(zip(acc1,acc2,samples_per_bin1,samples_per_bin2)):
            if s1==0 and s2==0:
                acc[idx]=0.0
            else:
                a1,a2,s1,s2=a1.float(),a2.float(),s1.float(),s2.float()
                acc[idx]=(lam*a1*s1+(1-lam)*a2*s2)/float(lam*s1+(1-lam)*s2)
            
        avg_conf,prob,samples_per_bin=average_confidence_per_bin(o,n_bins) 
        conf_per_bins,prob,samples_per_bin=confidence_per_bin(o,n_bins) 

        for a,c,c_per_bin,s in zip(acc,avg_conf,conf_per_bins,samples_per_bin):
            if s!=0.0:
                aux_cost+= calib_cost(a,c,c_per_bin,s, 2)

        tot_samples = float(sum(samples_per_bin))
        if tot_samples!=0:		
            aux_cost*=1/tot_samples

        COST+=aux_cost

    COST*= 28/float(len(bins_for_train))
    return COST
    
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
        self.mixup_coeff = 0.4
        self.bins_for_train = 10

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
# def cost_CONF_mixup_interpolated(o,t1,t2,lam):# \loss is applied over the mixup Image. 
#     COST=None
#     acc1=(o.argmax(dim=1)==t1).sum()#/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
#     acc2=(o.argmax(dim=1)==t2).sum()#/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
#     acc=(lam*acc1+(1-lam)*acc2)/float(t1.size(0))

#     predicted_prob=nn.functional.softmax(o,dim=1)
#     predicted_prob,index = torch.max(predicted_prob,1)

#     calib_cost_index = 1
#     if calib_cost_index==1:
#         COST=((predicted_prob.mean()-acc)**2)	
#     elif calib_cost_index==2:
#         COST=((predicted_prob-acc)**2).mean()

#     lamda = 1
#     return lamda*COST

# class MixupTrainer():
#     def __init__(self, **kwargs):
#         self.train_loader = kwargs['train_loader']
#         self.valid_loader = kwargs['valid_loader']
#         self.train_loader2 = kwargs['out_loader']

#         self.model =  kwargs['model']
#         self.optimizer =  kwargs['optimizer']
#         self.saver =  kwargs['saver']
#         self.scheduler =  kwargs['scheduler']
#         self.device =  kwargs['device']

#         self.criterion = nn.CrossEntropyLoss()
#         self.mixup_coeff = 0.4

#     def train(self):
#         self.model.train()
#         total_loss = 0
#         total = 0
#         correct = 0
#         for batch_idx, (data1, data2) in enumerate(zip(self.train_loader, self.train_loader2)):
#             inputs1, targets1 = data1
#             inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)

#             inputs2, targets2 = data2
#             inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

#             lam = np.random.beta(self.mixup_coeff, self.mixup_coeff)
#             inputs = lam * inputs1 + (1-lam) * inputs2

#             self.optimizer.zero_grad()

#             outputs = self.model(inputs)
#             loss = lam * self.criterion(outputs, targets1) + (1-lam) * self.criterion(outputs, targets2)

#             # calibration loss  
#             loss += cost_CONF_mixup_interpolated(outputs,targets1,targets2,lam) 


#             loss.backward()        
                
#             self.optimizer.step()

#             total_loss += loss
#             total += targets1.size(0)
#             _, predicted = outputs.max(1)            
#             correct += predicted.eq(targets1).sum().item()            
#             print('\r', batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                         % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')    

#         self.train_accuracy = correct/total
#         self.train_avg_loss = total_loss/len(self.train_loader)
#         print()

#     def validation(self, epoch):
#         self.model.eval()
#         total = 0
#         correct = 0
#         for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)

#             outputs = self.model(inputs)
#             total += targets.size(0)
#             _, predicted = outputs.max(1)  
#             correct += predicted.eq(targets).sum().item()           
#         valid_accuracy = correct/total
#         self.scheduler.step()
#         self.saver.save_checkpoint(epoch, metric = valid_accuracy)
#         print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, self.train_avg_loss, self.train_accuracy, valid_accuracy))
#         print(self.scheduler.get_last_lr())