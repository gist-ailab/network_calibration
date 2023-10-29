#
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


def get_soft_avuc_loss(probabilities, labels, soft_avuc_use_deprecated_v0,
                       soft_avuc_temp, soft_avuc_theta):

#   accuracies = tf.dtypes.cast(
#       tf.math.equal(
#           tf.argmax(probabilities, axis=1),
#           tf.cast(labels, tf.int64),
#       ),
#       tf.float32,
#   )
    accuracies = torch.eq(torch.max(probabilities, dim=1).index, labels).float()

    dim = int(probabilities.shape[1])
    uniform_probabilities = torch.tensor([1.0 / dim] * dim)
    log_safe_probabilities = (1.0 -
                                EPS) * probabilities + EPS * uniform_probabilities
    log_probabilities = torch.log(log_safe_probabilities)
    entropies = -(
        torch.sum(
            torch.multiply(log_safe_probabilities, log_probabilities), dim=1))

    entmax = torch.log(dim)

    # pylint: disable=g-long-lambda
    def soft_uncertainty(e, temp=1, theta=0.5):
        return torch.sigmoid(
            (1 / temp) * torch.log(e * (1 - theta) / ((1 - e) * theta)))

    xus = torch.map_fn(
        elems=entropies,
        fn=lambda ent: -((ent - entmax)**2),
    )
    xcs = torch.map_fn(
        elems=entropies,
        fn=lambda ent: -(ent**2),
    )
    qucs = torch.softmax(torch.stack([xus, xcs], dim=1), dim=1)
    qus = torch.squeeze(torch.slice(qucs, [0, 0], [-1, 1]))
    qcs = torch.squeeze(torch.slice(qucs, [0, 1], [-1, 1]))
    # pylint: enable=g-long-lambda

    accuracies_entropies_and_qucs = torch.stack([accuracies, entropies, qus, qcs],
                                            dim=1)

    # pylint: disable=g-long-lambda
    nac_diff = tf.reduce_sum(
        tf.map_fn(
            elems=accuracies_entropies_and_qucs,
            fn=lambda e: tf.convert_to_tensor(e[3]) *
            (tf.constant(1.0) - tf.math.tanh(tf.convert_to_tensor(e[1])))
            if e[0] > 0.5 else 0.0))
    nau_diff = tf.reduce_sum(
        tf.map_fn(
            elems=accuracies_entropies_and_qucs,
            fn=lambda e: tf.convert_to_tensor(e[2]) * tf.math.tanh(
                tf.convert_to_tensor(e[1])) if e[0] > 0.5 else 0.0))
    nic_diff = tf.reduce_sum(
        tf.map_fn(
            elems=accuracies_entropies_and_qucs,
            fn=lambda e: tf.convert_to_tensor(e[3]) *
            (tf.constant(1.0) - tf.math.tanh(tf.convert_to_tensor(e[1])))
            if e[0] < 0.5 else 0.0))
    niu_diff = tf.reduce_sum(
        tf.map_fn(
            elems=accuracies_entropies_and_qucs,
            fn=lambda e: tf.convert_to_tensor(e[2]) * tf.math.tanh(
                tf.convert_to_tensor(e[1])) if e[0] < 0.5 else 0.0))
    # pylint: enable=g-long-lambda

    avuc_loss = tf.math.log(
        tf.constant(1.0) + (nau_diff + nic_diff) /
        tf.math.maximum(nac_diff + niu_diff, tf.constant(EPS)))

    return avuc_loss


class Trainer():
    def __init__(self, **kwargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model
        self.optimizer = optimizer
        self.saver = saver
        self.scheduler = scheduler
        self.device = device

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
            outputs = self.model(inputs)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()           
        valid_accuracy = correct/total
        self.scheduler.step()
        self.saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, self.train_avg_loss, self.train_accuracy, valid_accuracy))
        print(self.scheduler.get_last_lr())
