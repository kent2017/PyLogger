import math


class LearningRateScheduler:
    @staticmethod
    def StepDecay(n, factor=0.5, min_lr=0):
        """
        Update lr every n epoches by a factor.
        @param n:
        @param min_lr: the minimum of the updated learning rate, 0 by default.
        """
        def update(epoch, lr):
            # epoch >= 0
            return max(min_lr, lr * factor**(epoch//n))
        return update

    @staticmethod
    def CustomDecay(epoch_lr_pairs:list):
        """
        @param epoch_lr_pairs: a list of pairs of (epoch, lr). Learing rate will be updated at the beginning of the
            corresponding epoch.
        """
        def update(epoch, lr):
            for e, _lr in epoch_lr_pairs[::-1]:
                if epoch >= e:
                    return _lr
        return update

    @staticmethod
    def TriangularCLR(epochsize, base_lr, max_lr):
        """
        Triangular cyclical learning rate. the max_lr will decay by 0.5 after each cycle.
            /\
           /  \
          /    \
         /      \
         ----
          ||
         epochsize
        @param epochsize: the number of epoches in half a cycle
        """
        def update(epoch, lr):
            cycle = math.floor(epoch//(2*epochsize))
            x = epoch/epochsize - 2*cycle
            scale = 1/(2.**cycle)
            return base_lr + (max_lr - base_lr) * min(x, 2-x) * scale
        return update

