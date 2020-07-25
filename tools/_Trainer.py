"""
_Trainer.py
@author kent
@date 2020/7/25
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

class BaseTrainer:
    """
    A base class of trainer.
    @param loss_fn: the loss function, must be the static function the of Loss class
    @param metrics: list, a list of metrics which must be of the Metric class
    @param log_dir: str, the logger directory, None by default.
    @param use_weights: bool, if use weights in the metrics.
    """

    def __init__(self, model:nn.Module,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 optimizer,
                 loss_fn,
                 metrics:list,
                 log_dir=None,
                 use_weights=False,
                 cuda=True):
        self.epoch = 1
        self.batch = 0

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.log_dir = log_dir
        self.use_weights = use_weights
        self.cuda = cuda

        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, end_epoch, start_epoch=1):
        """
        Train the model for (end_epoch-start_epoch+1) epoches.
        """
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.epoch = start_epoch

        self.on_train_begin()

        for epoch in range(start_epoch, end_epoch+1):
            self.model.train()
            self.epoch = epoch
            self.optimizer.zero_grad()

            self.on_epoch_begin(epoch)

            self.on_epoch_end(epoch)

        self.on_train_end()

    def test(self):
        self.model.eval()
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, logs=None):
        pass

    def on_epoch_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_train_batch_begin(self, logs=None):
        pass

    def on_train_batch_end(self, logs=None):
        pass

    def on_test_batch_begin(self, logs=None):
        pass

    def on_test_batch_end(self, logs=None):
        pass


if __name__ == "__main__":
    class T(BaseTrainer):
        def on_train_begin(self, logs=None):
            print("train begin")
        def on_train_end(self, logs=None):
            print("train end")
        def on_epoch_begin(self, epoch, logs=None):
            print("epoch begin")
        def on_epoch_end(self, epoch, logs=None):
            print("epoch end")

    t = T(None, None, None, None, None, None)
    t.train(10)