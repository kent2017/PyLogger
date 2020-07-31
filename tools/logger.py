"""
logger.py

@author kaidi
@date 2020/07/31
"""
import os
import  time
from tensorboardX import SummaryWriter


class Logger:
    """
    Log events under the log_dir.
    - lr, loss, metrics

    @param log_dir:
    @param args: arguments to be saved
    """
    def __init__(self, log_dir, args=None):
        log_dir = os.path.join(log_dir, time.strftime("Log_%Y-%m-%d_%H-%M-%S"))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

        # save
        if args:
            self._save_args(args)

        # states throughout the training phase
        self.epoch = 0              # current epoch
        self.total_epoches = 0      #
        self.step_acc = 0           # accumulated steps since the first epoch, will be added by one at the end of each step
        self.lr = 0.
        self.metric_names = []

        # states in the current epoch
        self.step = 0               # step in the current epoch
        self.loss_acc = 0.          # accumulated loss in the current epoch
        self.metric_values_acc = []

        self.val_loss_acc = 0.          # val loss
        self.val_metric_values_acc = []

        # states in the current step
        self.loss = 0.
        self.metric_values = []

        self.val_loss = 0.
        self.val_metric_values = []

        # files
        self.writer = SummaryWriter(log_dir)
        self.training_file = open(os.path.join(log_dir, "training.txt"), 'w')
        self.val_file = open(os.path.join(log_dir, "val.txt"), 'w')

        # file states
        self.first_write_training_file = True
        self.first_write_val_file = True

    def __del__(self):
        self.writer.close()
        self.training_file.close()
        self.val_file.close()

    def print_training(self, tbar=None, tbar_update=1):
        """
        Print logs on screen and write to a file.
        """
        epoch = self.epoch
        step = self.step

        metric_names = self.metric_names
        metric_values = self.metric_values_acc

        # 1. print on screen
        log_str = ' ;'.join(["%s %.3f" % (name, v) for name, v in zip(metric_names, metric_values)])
        log_str = 'epoch %d, step %d: loss %.3f; %s' % (epoch+1, step+1, self.loss_acc, log_str)

        if tbar:
            tbar.set_description(log_str)
            tbar.update(tbar_update)
        else:
            print(log_str)

        # 2. write to training file
        if self.first_write_training_file:
            print("epoch step loss %s"%(" ".join(metric_names)), file=self.training_file)
            self.first_write_training_file = False

        log_str = ' '.join(["%.3f"%v for v in metric_values])
        log_str = '%d %d %.3f %s' % (epoch+1, step+1, self.loss_acc, log_str)
        print(log_str, file=self.training_file)

    def print_val(self):
        epoch = self.epoch

        metric_names = ["val_%s"%name for name in self.metric_names]
        metric_values = self.val_metric_values_acc

        # 1. print on screen
        log_str = ' ;'.join(["%s %.3f" % (name, v) for name, v in zip(metric_names, metric_values)])
        log_str = 'epoch %d: val_loss %.3f; %s' % (epoch+1, self.val_loss_acc, log_str)
        print(log_str)

        # 2. write to val file
        if self.first_write_val_file:
            print("epoch val_loss %s"%(" ".join(metric_names)), file=self.val_file)
            self.first_write_val_file = False

        log_str = ' '.join(["%.3f"%v for v in metric_values])
        log_str = '%d %.3f %s' % (epoch+1, self.val_loss_acc, log_str)
        print(log_str, file=self.val_file)

    def write_summary_loss_metrics(self, train=True):
        if train:
            metric_names = self.metric_names
            metric_values = self.metric_values_acc

            self.writer.add_scalar('loss/train', self.loss_acc, self.step_acc+1)
            for name, v in zip(metric_names, metric_values):
                self.writer.add_scalar('%s/train'%(name), v, self.step_acc+1)
        else:
            # test
            metric_names = ["val_%s"%name for name in self.metric_names]
            metric_values = self.val_metric_values_acc
            for name, v in zip(metric_names, metric_values):
                self.writer.add_scalar('%s/test'%(name), v, self.epoch+1)

    def write_summary_params(self, train=True):
        epoch = self.epoch

        if train:
            self.writer.add_scalar('params/lr', self.lr, epoch+1)
        else:
            pass

    def _save_args(self, args):
        import json
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            s = json.dumps(vars(args), indent=2)
            print(s, file=f)
