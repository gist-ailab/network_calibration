import torch
from torch import nn, optim
from torch.nn import functional as F

from .common import _ECELoss, evaluate

import numpy as np

from .spline_util import *



class SplineEvaluater(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, **kwargs):
        super(SplineEvaluater, self).__init__()
        self.train_loader = kwargs['train_loader']
        self.valid_loader = kwargs['valid_loader']
        self.test_loader = kwargs['test_loader']


        self.model = kwargs['model']

        self.vector = nn.Parameter(torch.ones(kwargs['num_classes']) * 1.5)
        # self.bias = nn.Parameter(torch.ones(1) * 1.5)

        self.device = kwargs['device']

    def forward(self, input):
        logits = self.model(input)
        scores = torch.softmax(logits, dim=1)
        return self.spline_scale(scores)

    def spline_scale(self, scores_):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        scores_ = scores_.cpu().numpy()
        probs = scores_.max(1)
        scores = np.array([self.frecal(float(sc)) for sc in probs])
        scores[scores<0.0] = 0.0
        scores[scores>1.0] = 1.0
        return (scores_, scores)

    # This function probably should live outside of this class, but whatever
    def set_splines(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(self.device)
        self.model.eval()

        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        scores_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
                scores_list.append(torch.softmax(logits, dim=1))
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
            scores = torch.cat(scores_list).to(self.device)


        # Calculate NLL and ECE before vector scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()

        scores_, labels_, scores_class = get_top_results(scores.cpu().numpy(), labels, -1, return_topn_classid=True)
        self.frecal = get_recalibration_function(scores_, labels_, 'natural', 6)

        # Calculate NLL and ECE after temperature scaling
        scores_ = np.array([self.frecal(float(sc)) for sc in scores_])
        scores_[scores_<0.0] = 0.0
        scores_[scores_>1.0] = 1.0
        
        after_temperature_ece = ece_criterion(self.spline_scale(scores), labels, True).item()
        print('After temperature - ECE: %.3f' % (after_temperature_ece))

        return self


    def eval(self):
        self.model.eval()
        self.set_splines(self.valid_loader)

        outputs = []
        targets = []
        splines = []
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(self.test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.forward(inputs)
                # prob, _ = output.max(1) 

                outputs.append(output[0])
                splines.append(output[1])
                targets.append(target)
        outputs = np.concatenate(outputs)
        splines = np.concatenate(splines)
        targets = torch.cat(targets)
        evaluate([outputs, splines], targets.cpu().numpy(), verbose = True, is_spline=True) #probs, y_true