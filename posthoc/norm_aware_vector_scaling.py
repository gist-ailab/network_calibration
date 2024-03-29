import torch
from torch import nn, optim
from torch.nn import functional as F

from .common import _ECELoss, evaluate


class VectorNormEvaluater(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, **kwargs):
        super(VectorNormEvaluater, self).__init__()
        self.train_loader = kwargs['train_loader']
        self.valid_loader = kwargs['valid_loader']
        self.test_loader = kwargs['test_loader']

        self.model = kwargs['model']
        self.ndim = kwargs['ndim']
        self.vector = nn.Parameter(torch.ones(kwargs['num_classes']) * 1.5)
        self.norm_temperature = nn.Parameter(torch.ones(1) * 1.5)

        self.device = kwargs['device']

    def forward(self, input):
        features = self.model.forward_features(input)
        features = F.adaptive_avg_pool2d(features, [1,1]).view(-1, self.ndim)
        return self.norm_scale(features)

    def norm_scale(self, features):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        norm = torch.norm(features, p=1, dim=1, keepdim=True)
        f_max = torch.max(features, dim=1, keepdim=True).values
        norm_temperature = self.norm_temperature.unsqueeze(1).expand(features.size(0), features.size(1))
        features = features * norm_temperature * (norm/f_max)

        logits = self.model.fc(features)
        vector = self.vector.unsqueeze(0)
        return logits * vector

    # This function probably should live outside of this class, but whatever
    def set_norm_scale(self, valid_loader):
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
        features_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                features = self.model.forward_features(input)
                features = F.adaptive_avg_pool2d(features, [1,1]).view(-1, self.ndim)
                logits = self.model.fc(features)

                features_list.append(features)
                logits_list.append(logits)
                labels_list.append(label)
            features = torch.cat(features_list).to(self.device)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before norm - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.norm_temperature, self.vector], lr=0.001, max_iter=5000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.norm_scale(features), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.norm_scale(features), labels).item()
        after_temperature_ece = ece_criterion(self.norm_scale(features), labels).item()
        print('Optimal temperature: {} {:.3f}'.format(self.vector.cpu().detach().numpy(), self.norm_temperature.item()))
        print('After norm - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


    def eval(self):
        self.model.eval()
        self.set_norm_scale(self.valid_loader)

        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(self.test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.forward(inputs)
                output = torch.softmax(output, dim=1)
                # prob, _ = output.max(1) 

                outputs.append(output)
                targets.append(target)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        evaluate(outputs.cpu().numpy(), targets.cpu().numpy(), verbose = True) #probs, y_true

