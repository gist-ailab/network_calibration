import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import evaluate

class BaselineEvaluater():
    def __init__(self, **kwargs):
        self.train_loader = kwargs['train_loader']
        self.valid_loader = kwargs['valid_loader']
        self.test_loader = kwargs['test_loader']

        self.model =  kwargs['model']
        self.device =  kwargs['device']

    def eval(self):
        self.model.eval()

        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(self.test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                output = torch.softmax(output, dim=1)
                # prob, _ = output.max(1) 

                outputs.append(output)
                targets.append(target)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        evaluate(outputs.cpu().numpy(), targets.cpu().numpy(), verbose = True) #probs, y_true

