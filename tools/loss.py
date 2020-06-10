import torch

_epsilon = 1e-12

class Loss:
    Tensor = torch.Tensor

    @staticmethod
    def JaccardBCELoss(output, target):
        # type: (Tensor, Tensor) -> Tensor
        """
        @param output: (N, 1, W, H)
        @param target: (N, 1, W, H)
        @return loss: scalar
        """
        assert output.shape==target.shape, "input shape must be the same as target shape"
        assert torch.all((output>=0.) & (output <=1.)), "input value must be in [0., 1.]"
        assert torch.all((target>=0.) & (target<=1.)), "target value must be in [0., 1.]"

        output = output.float()
        target = target.float()

        axis = [i for i in range(1, output.ndimension())]
        inter = torch.sum(output*target, axis)
        union = torch.sum(output+target, axis) - inter

        iou = (inter+_epsilon)/(union+_epsilon)     # (n, )
        iou = torch.clamp(iou, _epsilon, 1.)

        loss = -torch.log(iou).mean() + Loss.BCELoss(output, target)
        return loss

    @staticmethod
    def BCELoss(output, target):
        # type: (Tensor, Tensor) -> Tensor
        """
        A criterion that measures the binary cross entropy between the input and the target.
        @param output: shape (N, 1) or (N, 1, H, W)
        @param target: shape (N, 1) or (N, 1, H, W)
        @return loss: scalar
        """
        assert output.shape==target.shape, "input shape must be the same as target shape"
        assert torch.all((output>=0.) & (output <=1.)), "input value must be in [0., 1.]"
        assert torch.all((target>=0.) & (target<=1.)), "target value must be in [0., 1.]"

        output = output.float()
        target = target.float()

        axis = [i for i in range(1, output.ndimension())]

        bce = -(target * torch.log(torch.clamp(output, _epsilon, 1.)) + (-target + 1.)*torch.log(torch.clamp(-output + 1., _epsilon, 1.))).mean(axis)    # (n,)
        loss = bce.mean()
        return loss


if __name__ == "__main__":
    output = torch.empty((2, 1, 2, 1), dtype=torch.float).random_(0, 101) / 100.
    target = torch.empty((2, 1, 2, 1), dtype=torch.float).random_(0, 2)
    res = Loss.sigmoid_jaccard_loss(output, target)
    print(res)
    print(torch.nn.BCELoss()(output, target).item())
    pass

