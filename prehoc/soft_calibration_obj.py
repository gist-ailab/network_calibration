import torch
import torch.nn as nn
import torch.nn.functional as F

def get_soft_avuc_loss(probabilities, labels, soft_avuc_use_deprecated_v0,
                       soft_avuc_temp, soft_avuc_theta):
  """Computes and returns the soft AvUC loss tensor.
  Soft AvUC loss is defined in equation (15) in this paper:
  https://arxiv.org/pdf/2108.00106.pdf.
  Args:
    probabilities: tensor of predicted probabilities of
      (batch-size, num-classes) shape
    labels: tensor of labels in [0,num-classes) of (batch-size,) shape
    soft_avuc_use_deprecated_v0: whether to use a deprecated formulation
    soft_avuc_temp: temperature > 0 (T in equation (15) cited above)
    soft_avuc_theta: threshold in (0,1) (kappa in equation (15) cited above)
  Returns:
    A tensor of () shape containing a single value: the soft AvUC loss.
  """

  accuracies = tf.dtypes.cast(
      tf.math.equal(
          tf.argmax(probabilities, axis=1),
          tf.cast(labels, tf.int64),
      ),
      tf.float32,
  )

  dim = int(probabilities.shape[1])
  uniform_probabilities = tf.convert_to_tensor([1.0 / dim] * dim)
  log_safe_probabilities = (1.0 -
                            EPS) * probabilities + EPS * uniform_probabilities
  log_probabilities = tf.math.log(log_safe_probabilities)
  entropies = tf.math.negative(
      tf.reduce_sum(
          tf.multiply(log_safe_probabilities, log_probabilities), axis=1))

  entmax = math.log(dim)

  # pylint: disable=g-long-lambda
  def soft_uncertainty(e, temp=1, theta=0.5):
    return tf.math.sigmoid(
        (1 / temp) * tf.math.log(e * (1 - theta) / ((1 - e) * theta)))

  if soft_avuc_use_deprecated_v0:
    xus = tf.map_fn(
        elems=entropies,
        fn=lambda ent: -((ent - entmax)**2),
    )
    xcs = tf.map_fn(
        elems=entropies,
        fn=lambda ent: -(ent**2),
    )
    qucs = tf.nn.softmax(tf.stack([xus, xcs], axis=1), axis=1)
    qus = tf.squeeze(tf.slice(qucs, [0, 0], [-1, 1]))
    qcs = tf.squeeze(tf.slice(qucs, [0, 1], [-1, 1]))
  else:
    qus = tf.map_fn(
        elems=entropies,
        fn=lambda ent: soft_uncertainty(
            ent / entmax, temp=soft_avuc_temp, theta=soft_avuc_theta),
    )
    qcs = tf.map_fn(
        elems=qus,
        fn=lambda qu: 1 - qu,
    )
  # pylint: enable=g-long-lambda

  accuracies_entropies_and_qucs = tf.stack([accuracies, entropies, qus, qcs],
                                           axis=1)

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