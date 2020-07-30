

class LearningRateScheduler:
    @staticmethod
    def StepDecay(n, factor=0.5):
        """
        Update lr every n epoches by a factor.
        @param n:
        """
        def update(epoch, lr):
            lr *= factor**(epoch//n)
            return lr
        return update
