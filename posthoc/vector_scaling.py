import torch
from torch import nn, optim
from torch.nn import functional as F

from .common import _ECELoss


class ModelWithVectorScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithVectorScaling, self).__init__()
        self.model = model
        self.w = nn.Parameter(torch.ones(self.model.num_classes) * 1.5)
        self.b = nn.Parameter(torch.ones(self.model.num_classes) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.vector_scale(logits)

    def vector_scale(self, logits):
        """
        Perform vector scaling on logits
        """
        # Expand temperature to match the size of logits
        
        logits = self.w * logits + self.b
        return logits

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.vector_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.vector_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.vector_scale(logits), labels).item()
        print('Optimal vector scaling W: %.3f, b: %.3f' % self.w.item(), self.b.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


