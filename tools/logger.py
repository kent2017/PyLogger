"""
logger.py

@author kaidi
@date 2020/07/31
"""
import os
import time
from tensorboardX import SummaryWriter


class Logger:
    """
    Log events under the log_dir.
    - lr, loss, metrics

    @param log_dir:
    @param args: arguments to be saved
    """
    def __init__(self, log_dir, args=None):
        self.log_dir = log_dir

        # info
        self.batch_data_info = None

        # states throughout the training phase
        self.epoch = 0              # current epoch
        self.total_epoches = 0      #
        self.step_acc = 0           # accumulated steps since the first epoch, will be added by one at the end of each step
        self.lr = 0.
        self.metric_names = []
        self.loss_names = []        # names of losses

        # states in the current epoch
        self.step = 0               # step in the current epoch
        self.loss_acc = 0.          # accumulated total loss in the current epoch
        self.losses = []            # values of losses
        self.metric_values_acc = []

        self.val_step = 0
        self.val_loss_acc = 0.          # val loss
        self.val_losses = []
        self.val_metric_values_acc = []

        # states in the current step
        self.loss = 0.
        self.metric_values = []

        self.val_loss = 0.
        self.val_metric_values = []

        # files
        self.writer = SummaryWriter(log_dir)
        self.log_file = open(os.path.join(log_dir, "log.txt"), 'w')
        self.val_file = open(os.path.join(log_dir, "val.txt"), 'w')

    def __del__(self):
        self.writer.close()
        self.log_file.close()
        self.val_file.close()

    def print_training(self, tbar=None, tbar_update=1):
        """
        Print logs on screen and write to a file.
        """
        epoch = self.epoch
        step = self.step

        names = [*self.loss_names, *self.metric_names]
        values = [*self.losses, *self.metric_values_acc]

        log_str = '; '.join(["%s %.4f" % (name, v) for name, v in zip(names, values)])
        log_str = 'epoch %d, step %d: loss %.4f; %s' % (epoch, step, self.loss_acc, log_str)

        if tbar:
            tbar.set_description(log_str)
            tbar.update(tbar_update)
            print(log_str, file=self.log_file)
        else:
            print(log_str)

    def print_val(self):
        epoch = self.epoch

        names = [*["val_%s"%name for name in self.loss_names], *["val_%s"%name for name in self.metric_names]]
        values = [*self.val_losses, *self.val_metric_values_acc]

        log_str = '; '.join(["%s %.4f" % (name, v) for name, v in zip(names, values)])
        log_str = 'epoch %d: val_loss %.4f; %s' % (epoch, self.val_loss_acc, log_str)
        self.log(log_str)
        print(log_str, file=self.val_file, flush=True)

    def write_summary_loss_metrics(self, train=True):
        if train:
            metric_names = self.metric_names
            metric_values = self.metric_values_acc

            self.writer.add_scalar('loss/train', self.loss_acc, self.step_acc)
            for name, v in zip(metric_names, metric_values):
                self.writer.add_scalar('%s/train'%(name), v, self.step_acc)
        else:
            # test
            metric_names = self.metric_names
            metric_values = self.val_metric_values_acc

            self.writer.add_scalar('loss/test', self.val_loss_acc, self.epoch)
            for name, v in zip(metric_names, metric_values):
                self.writer.add_scalar('%s/test'%(name), v, self.epoch)

    def write_summary_params(self, train=True):
        epoch = self.epoch

        if train:
            self.writer.add_scalar('params/lr', self.lr, epoch)
        else:
            pass

    def _save_args(self, args):
        import platform
        import json
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            config = vars(args)
            config['platform'] = platform.node()
            s = json.dumps(config, indent=2)
            print(s, file=f)

    def log(self, message:str):
        print(message)
        print(message, file=self.log_file, flush=True)
