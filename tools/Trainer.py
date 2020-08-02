"""
Trainer.py
@author kent
@date 2020/7/25
"""
import torch
import os
import time
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from .logger import Logger


class BaseTrainer:
    """
    A base class of trainer.
    @param loss_fn: the loss function, must be the static function the of Loss class
    @param metrics: list, a list of metrics which must be of the Metric class
    @param use_weights: bool, if use weights in the metrics.
    """

    def __init__(self, model:nn.Module,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 optimizer,
                 loss_fn,
                 metrics:list,
                 use_weights=False,
                 lr_scheduler=None,
                 cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.use_weights = use_weights
        self.lr_scheduler = lr_scheduler
        self.cuda = cuda

        if self.cuda:
            self.model = self.model.cuda()

    def train(self, epoches,
              logger:Logger,
              start_epoch=0,
              val_epoches=1,
              print_steps=10,
              write_summary_epoch=None,
              write_summary_steps=10,
              save_weights_epoch=1):
        """
        Train the model for (end_epoch-start_epoch) epoches.
        @param epoches: the number of epoches throughout the training phase
        @param start_epoch: the start training epoch
        @param val_epoches: the number of epoches between two validatings.
        @param print_steps: the number of steps between two printings
        @param write_summary_epoch: the number of epoches between two summary writings.
                    If specified, this will override write_summary_steps.
        @param write_summary_steps: the number of steps between two summary writings.
        @param save_weights_epoch: the number of epoches between two weights savings
        """
        if write_summary_epoch:
            write_summary_steps=None

        # global states
        logger.epoch = start_epoch
        logger.total_epoches = epoches
        logger.step_acc = 0
        logger.metric_names = [m.name for m in self.metrics]
        self.lr = self.optimizer.defaults['lr']

        self.on_train_begin(logger)       # customize

        for epoch in range(start_epoch, epoches):
            # 1. epoch begin
            #  lr scheduler
            if self.lr_scheduler:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_scheduler(epoch, self.optimizer.defaults['lr'])     # epoch is from 0

            # epoch states
            logger.epoch = epoch
            logger.lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            logger.step = 0
            logger.loss_acc = 0.
            logger.metric_values_acc = [0.]*len(self.metrics)
            logger.metric_values = [0.]*len(self.metrics)

            logger.write_summary_params(True)

            # zero the accumulated values
            for m in self.metrics:
                m.zero_values()     # zero the accumulated metric values

            self.on_epoch_begin(logger)   # customize

            batch_bar = tqdm(range(len(self.train_loader)))

            # 2. train in batches
            for step, data in enumerate(self.train_loader):
                # logger states
                logger.step = step

                self.on_train_batch_begin(logger)     # customize

                self.in_train_batch(logger, data)

                # print logs
                if (step+1)%print_steps==0:
                    logger.print_training(batch_bar, print_steps)
                elif step == len(self.train_loader) -1:
                    logger.print_training(batch_bar, len(self.train_loader)%print_steps)

                # write summary
                if write_summary_steps and (logger.step_acc+1)%write_summary_steps==0:
                    logger.write_summary_loss_metrics(train=True)

                self.on_train_batch_end(logger)       # customize
                logger.step_acc += 1

            if write_summary_epoch and (epoch + 1) % write_summary_epoch == 0:
                logger.write_summary_loss_metrics(train=True)
            batch_bar.close()

            # 3. test
            if (epoch+1)%val_epoches==0:
                self.test(logger)
                logger.print_val()
                logger.write_summary_loss_metrics(train=False)
                print("")

            # 4. save weights
            if epoch == epoches-1:
                self.save(logger, optimizer=True)
            elif (epoch+1)%save_weights_epoch == 0:
                self.save(logger)

            self.on_epoch_end(logger)

        self.on_train_end(logger)

    def test(self, logger:Logger):
        self.on_test_begin(logger)

        # zero values
        logger.val_loss_acc = 0.
        logger.val_metric_values = [0.] * len(self.metrics)
        logger.val_metric_values_acc = [0.] * len(self.metrics)
        for m in self.metrics:
            m.zero_values()

        self.model.eval()
        with torch.no_grad():
            # test
            for step, data in tqdm(enumerate(self.val_loader, 0)):
                logger.val_step = step

                self.on_test_batch_begin(logger)
                self.in_test_batch(logger, data)
                self.on_test_batch_end(logger)
        self.on_test_end(logger)

    def in_train_batch(self, logger:Logger, data):
        """
        @param step: step in the current epoch
        """
        device = 'cuda' if self.cuda else 'cpu'

        self.model.train()
        self.optimizer.zero_grad()

        step = logger.step

        # 1). input and output
        in_data = data[0].to(device).float()
        target = data[1].to(device).long()
        info = data[2]

        out_data = self.model(in_data)

        logger.batch_data_info = info
        # 2). loss
        if self.use_weights:
            weights = 1
            loss = self.loss_fn(out_data, target, weights)
        else:
            loss = self.loss_fn(out_data, target)

        logger.loss = loss.item()
        logger.loss_acc += (loss.item()-logger.loss_acc) / (step+1.)

        # 3). metrics
        for j, metric in enumerate(self.metrics):
            if self.use_weights:
                weights = 1     # customize
                v = metric.forward(out_data, target, weights)
            else:
                v = metric.forward(out_data, target)

            logger.metric_values[j] = v
            if metric.accumulated:
                logger.metric_values_acc[j] = v
            else:
                logger.metric_values_acc[j] += (v - logger.metric_values_acc[j]) / (step + 1.)  # avg metric values

        # 4). loss backward
        loss.backward()
        self.optimizer.step()

    def in_test_batch(self, logger:Logger, data):
        device = 'cuda' if self.cuda else 'cpu'
        # 1.output
        in_data = data[0].to(device).float()
        target = data[1].to(device).long()
        info = data[2]
        out_data = self.model(in_data)

        logger.batch_data_info = info

        # 2. loss
        if self.use_weights:
            weights = 1
            val_loss = self.loss_fn(out_data, target, weights).item()
        else:
            val_loss = self.loss_fn(out_data, target).item()

        logger.val_loss = val_loss
        logger.val_loss_acc += (logger.val_loss - logger.val_loss_acc) / (logger.val_step + 1.)

        # 3. metrics
        for j, metric in enumerate(self.metrics):
            if self.use_weights:
                weights = 1
                v = metric.forward(out_data, target, weights)
            else:
                v = metric.forward(out_data, target)

            logger.val_metric_values[j] = v
            if metric.accumulated:
                logger.val_metric_values_acc[j] = v
            else:
                logger.val_metric_values_acc[j] += (v - logger.val_metric_values_acc[j]) / (logger.val_step + 1.)  # avg metric values

    def on_train_begin(self, logger:Logger):
        pass

    def on_train_end(self, logger:Logger):
        pass

    def on_epoch_begin(self, logger:Logger):
        pass

    def on_epoch_end(self, logger:Logger):
        pass

    def on_test_begin(self, logger:Logger):
        pass

    def on_test_end(self, logger:Logger):
        pass

    def on_train_batch_begin(self, logger:Logger):
        pass

    def on_train_batch_end(self, logger:Logger):
        pass

    def on_test_batch_begin(self, logger:Logger):
        pass

    def on_test_batch_end(self, logger:Logger):
        pass

    def save(self, logger:Logger, optimizer=False):
        if optimizer:
            fn = os.path.join(logger.log_dir, "optimizer.t7")
            torch.save(self.optimizer.state_dict(), fn)
            print("Save optimizer to %s\n"%fn)

        dir_checkpoints = os.path.join(logger.log_dir, 'checkpoints')
        if not os.path.exists(dir_checkpoints):
            os.mkdir(dir_checkpoints)

        fn = os.path.join(dir_checkpoints, "%d_%s_%.4f.t7" % (logger.epoch, self.metrics[0].name, logger.val_metric_values_acc[0]))
        torch.save(self.model.state_dict(), fn)
        print("Save model to %s\n" % fn)

    def load_optimizer(self, fn):
        ck = torch.load(fn)
        self.optimizer.load_state_dict(ck)
        print("Load optimizer %s\n"%fn)
