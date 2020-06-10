import torch

_epsilon = 1e-12


class IOU:
    def __init__(self):
        self.name = 'iou'

    def forward(self, output:torch.Tensor, target:torch.Tensor):
        """
        @param input: (N, 1, H, W) or (N, 1)
        @param target: (N, 1, H, W) or (N, 1)
        @return mean_iou: scalar
        """
        assert torch.all((output>=0.) & (output <=1.)), "output value must be in [0., 1.]"
        assert torch.all((target>=0.) & (target<=1.)), "target value must be in [0., 1.]"

        if len(output.shape) == 2:
            return self._iou_class(output, target)
        elif len(output.shape) == 4:
            return self._iou_image(output, target)
        else:
            assert 0, "inapplicable"

    def _iou_image(self, output:torch.Tensor, target:torch.Tensor):
        """
        @param input: (N, 1, H, W)
        @param target: (N, 1, H, W)
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


class Accuracy:
    def __init__(self):
        self.name = 'acc'

    def forward(self, output:torch.Tensor, target:torch.Tensor):
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

if __name__ == "__main__":
    m = IOU()
    a = torch.randint(0, 2, (10, 1))
    b = torch.randint(0, 2, (10, 1))
    iou = m.forward(a, b)
    pass
