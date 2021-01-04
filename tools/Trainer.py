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
    @param loss_fns: list, a list of loss functions, must be the static function the of Loss class
    @param metrics: list, a list of metrics which must be of the Metric class
    """

    def __init__(self, model:nn.Module,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 optimizer,
                 loss_fns:dict,
                 metrics:list=[],
                 lr_scheduler=None,
                 cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fns = loss_fns
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.cuda = cuda

        if self.cuda:
            self.model = self.model.cuda()

    def train(self, epochs,
              logger:Logger,
              start_epoch=0,
              val_epochs=1,
              print_steps=10,
              write_summary_epoch=None,
              write_summary_steps=10,
              save_weights_epoch=1):
        """
        Train the model for (end_epoch-start_epoch) epochs.
        @param epochs: the number of epochs throughout the training phase
        @param start_epoch: the start training epoch
        @param val_epochs: the interval (epochs) between two validatings.
        @param print_steps: the interval (steps) between two printings
        @param write_summary_epoch: the interval (epochs) between two summary writings.
                    If specified, this will override write_summary_steps.
        @param write_summary_steps: the interval (steps) between two summary writings.
        @param save_weights_epoch: the interval (epochs) between two weights savings
        """
        if write_summary_epoch:
            write_summary_steps=None

        # global states
        logger.epoch = start_epoch
        logger.total_epochs = epochs
        logger.step_acc = 0
        logger.metric_names = [m.name for m in self.metrics]
        logger.loss_names = self.loss_fns.keys()
        self.lr = self.optimizer.defaults['lr']

        self.on_train_begin(logger)       # customize

        for epoch in range(start_epoch, epochs):
            # 1. epoch begin
            # epoch states
            now = time.strftime("%c")
            logger.log('\n================ Training Loss (%s) ================' % now)

            logger.epoch = epoch + 1
            logger.lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            logger.step = 0
            logger.loss_acc = 0.
            logger.metric_values_acc = [0.]*len(self.metrics)
            logger.metric_values = [0.]*len(self.metrics)
            logger.losses = [0.]*len(logger.loss_names)

            logger.write_summary_params(True)

            # zero the accumulated values
            for m in self.metrics:
                m.zero_values()     # zero the accumulated metric values

            self.on_epoch_begin(logger)   # customize

            batch_bar = tqdm(range(len(self.train_loader)))

            # 2. train in batches
            step_data_time = time.time()
            for step, data in enumerate(self.train_loader):
                step_start_time = time.time()
                # logger states
                logger.step = step

                self.on_train_batch_begin(logger)     # customize

                self.in_train_batch(logger, data)

                # print logs
                logger.data_time = step_start_time - step_data_time
                logger.runtime = time.time() - step_start_time
                if (step+1)%print_steps==0:
                    logger.print_training(batch_bar, print_steps)
                elif step == len(self.train_loader) -1:
                    logger.print_training(batch_bar, len(self.train_loader)%print_steps)

                # write summary
                if write_summary_steps and (logger.step_acc+1)%write_summary_steps==0:
                    logger.write_summary_loss_metrics(train=True)

                self.on_train_batch_end(logger)       # customize
                logger.step_acc += 1
                step_data_time = time.time()

            if write_summary_epoch and (epoch + 1) % write_summary_epoch == 0:
                logger.write_summary_loss_metrics(train=True)
            batch_bar.close()

            #  lr scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 3. test
            if (epoch+1)%val_epochs==0:
                self.test(logger)
                logger.print_val()
                logger.write_summary_loss_metrics(train=False)

            # 4. save weights
            if epoch == epochs-1:
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
        logger.val_losses = [0.] * len(logger.loss_names)
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
        x = data[0].to(device)
        y = data[1].to(device)

        out = self.model(x)

        # 2). loss
        loss = 0.
        for i, fn in enumerate(self.loss_fns.values()):
            _l = fn(out, y)
            loss = _l + loss
            logger.losses[i] = _l.item()

        logger.loss = loss.item()
        logger.loss_acc += (loss.item()-logger.loss_acc) / (step+1.)

        # 3). metrics
        for j, metric in enumerate(self.metrics):
            v = metric(out, y)

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
        x = data[0].to(device)
        y = data[1].to(device)

        out = self.model(x)

        # 2. loss
        val_loss = 0.
        for i, fn in enumerate(self.loss_fns.values()):
            _l = fn(out, y)
            val_loss = _l + val_loss
            logger.val_losses[i] = _l.item()

        logger.val_loss = val_loss.item()
        logger.val_loss_acc += (logger.val_loss - logger.val_loss_acc) / (logger.val_step + 1.)

        # 3. metrics
        for j, metric in enumerate(self.metrics):
            v = metric(out, y)

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
            logger.log("Save optimizer to %s\n"%fn)

        dir_checkpoints = os.path.join(logger.log_dir, 'checkpoints')
        if not os.path.exists(dir_checkpoints):
            os.mkdir(dir_checkpoints)

        # save
        fn = os.path.join(dir_checkpoints, "%d_%s_%.4f.t7" % (logger.epoch, 'loss', logger.val_loss_acc))
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), fn)
        else:
            torch.save(self.model.state_dict(), fn)
        logger.log("Save model to %s\n" % fn)

    def load_optimizer(self, fn):
        ck = torch.load(fn)
        self.optimizer.load_state_dict(ck)
        print("Load optimizer %s\n"%fn)
