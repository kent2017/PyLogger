import torch
import numpy as np
from tools.Metric import Metric

_epsilon = 1e-12


class IOU(Metric):

    def __init__(self):
        super(IOU, self).__init__(is_accumulated=False)
        self.name = 'iou'

    def forward(self, output:torch.Tensor, target:torch.Tensor, weights:torch.Tensor=None):
        """
        @param input: (N, C, H, W), (N, C, H) or (N, C), where C = 1 or 2
        @param target: (N, C, H, W), (N, C, H) or (N, C), where C = 1 or 2
                    or (N, H, W),    (N, H)    or (N, )
        @param weights: (N, H, W), or (N, H)
        @return mean_iou: scalar
        """
        n = output.size(0)
        if output.shape[1] > 1:
            output = output.argmax(1, keepdim=True)
        if len(target.shape) == len(output.shape) and target.shape[1] > 1:
            target = target.argmax(1, keepdim=True)

        # assert torch.all((output>=0.) & (output <=1.)), "output value must be in [0., 1.]"
        # assert torch.all((target>=0.) & (target<=1.)), "target value must be in [0., 1.]"

        output = output.contiguous()
        target = target.contiguous()

        if len(output.shape) == 2:
            return self._iou_class(output, target)
        elif len(output.shape) > 2:
            output = output.view(n, -1)
            target = target.view(n, -1)

            if weights is None:
                return self._iou_image(output, target)
            else:
                return self._iou_weighted(output, target, weights)
        else:
            assert 0, "inapplicable"

    def _iou_image(self, output:torch.Tensor, target:torch.Tensor):
        """
        @param input: (N, H, W)
        @param target: (N, H, W)
        @return mean_iou: scalar
        """
        output = output.float()
        target = target.float()

        axis = [i for i in range(1, output.ndimension())]
        inter = torch.sum(output*target, axis)
        union = torch.sum(output+target, axis) - inter

        iou = (inter+_epsilon)/(union+_epsilon)     # (n, )
        return iou.mean().item()

    def _iou_class(self, output:torch.Tensor, target:torch.Tensor):
        """
        @param input: (N, 1)
        @param target: (N, 1)
        @return mean_iou: scalar
        """
        output = output.float()
        target = target.float()

        inter = torch.sum(output*target, [0, 1])
        union = torch.sum(output+target, [0, 1]) - inter

        iou = (inter+_epsilon)/(union+_epsilon)
        return iou.item()

    def _iou_weighted(self, output:torch.Tensor, target:torch.Tensor, weights:torch.Tensor):
        """
        @param input: (N, H)
        @param target: (N, H)
        @param weights: (N, H)
        @return mean_iou: scalar
        """
        n = output.size(0)
        output = output.float()
        target = target.float()

        # weights = weights.float()
        # weights = weights.contiguous()
        # weights = weights.view(n, -1)  # (n, h)
        # weights = weights / weights.sum(1, keepdim=True) * float(weights.size(1))
        #
        axis = [i for i in range(1, output.ndimension())]
        inter = torch.sum(output*target*weights, axis)
        union = torch.sum((output+target)*weights, axis) - inter

        iou = (inter+_epsilon)/(union+_epsilon)     # (n, )
        return iou.mean().item()


class Accuracy(Metric):

    def __init__(self):
        super(Accuracy, self).__init__(is_accumulated=False)
        self.name = 'acc'

    def forward(self, output:torch.Tensor, target:torch.Tensor, weights=None):
        """
        @param output: (N, C, H, W) or (N, C), where C>=1
        @param target: (N, C, H, W) or (N, C), where C>=1
        @return mean_acc: scalar
        """
        assert torch.all((output>=0.) & (output <=1.)), "input value must be in [0., 1.]"
        assert torch.all((target>=0.) & (target<=1.)), "target value must be in [0., 1.]"

        if output.shape[1] == 1:
            return self._binary_acc(output, target)
        else:
            return self._multi_acc(output, target)

    def _binary_acc(self, output:torch.Tensor, target:torch.Tensor):
        """
        @param output: (N, 1, H, W) or (N, 1)
        @param target: (N, 1, H, W) or (N, 1)
        @return mean_acc: scalar
        """
        output = (output>0.5).int()
        target = (target>0.5).int()

        axis = [i for i in range(1, output.ndimension())]
        acc = torch.mean(output == target, dtype=torch.float, dim=axis)     #(n, )
        return acc.mean().item()

    def _multi_acc(self, output:torch.Tensor, target:torch.Tensor):
        """
        @param output: (N, C, H, W) or (N, C), where C>1
        @param target: (N, C, H, W) or (N, C)
        @return mean_acc: scalar
        """
        assert 0, "not implemented"


class Recall(Metric):
    """
    Measure the recall during the training and validation process. TP and P will be accumulated during the phrase.
    Use zero_values to zero TP and P at the beginning of the training phrase.
    """

    def __init__(self):
        super(Recall, self).__init__(is_accumulated=True)
        self.name = "recall"
        self.TP = 0
        self.P = 0      # actual positive

    def zero_values(self):
        self.TP = 0
        self.P = 0

    def forward(self, output:torch.Tensor, target:torch.Tensor, weights=None):
        """
        @param output: (N, 1)
        @param target: (N, 1)
        @return recall: scalar
        """
        assert len(output.shape)==2 and len(target.shape)==2
        assert output.shape[1] == 1 and target.shape[1] == 1

        output = (output>0.5).int()
        target = (target>0.5).int()

        TP = (output & target).sum().item()
        P = target.sum().item()
        self.TP += TP
        self.P += P

        recall = (self.TP + _epsilon) / (self.P + _epsilon)
        return recall


class Precision(Metric):
    """
    Measure the precision during the training and validation process. TP and P' will be accumulated during the phrase.
    Use zero_values to zero TP and P' at the beginning of the training phrase.
    """

    def __init__(self):
        super(Precision, self).__init__(is_accumulated=True)
        self.name = "precision"
        self.TP = 0
        self.P0 = 0      # predicted positive

    def zero_values(self):
        self.TP = 0
        self.P0 = 0

    def forward(self, output:torch.Tensor, target:torch.Tensor, weights=None):
        """
        @param output: (N, 1)
        @param target: (N, 1)
        @return recall: scalar
        """
        assert len(output.shape)==2 and len(target.shape)==2
        assert output.shape[1] == 1 and target.shape[1] == 1

        output = (output>0.5).int()
        target = (target>0.5).int()

        TP = (output & target).sum().item()
        P0 = output.sum().item()
        self.TP += TP
        self.P0 += P0

        recall = (self.TP + _epsilon) / (self.P0 + _epsilon)
        return recall


if __name__ == "__main__":
    precision = Precision()
    for i in range(3):
        precision.zero_values()
        for _ in range(5):
            a = torch.randint(0, 2, (5, 1))
            b = torch.randint(0, 2, (5, 1))
            print(precision.forward(a, b))
    pass
