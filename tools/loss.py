import torch
import torch.nn.functional as F

_epsilon = 1e-12

class Loss:
    Tensor = torch.Tensor

    @staticmethod
    def CrossEntropy(output, target, weights=None):
        # type:(Tensor, Tensor, Tensor) -> Tensor
        """
        softmax inside this function!
        @param output: (N, C, H, W), float
        @param target: (N, H, W), long
        @param weights: (N, H, W), float
        @return loss: scalar
        """
        n = output.size(0)
        c = output.size(1)

        output = output.contiguous()
        output = output.view(n, c, -1)      #(n, c, h)

        target = target.long()
        target = target.contiguous()
        target = target.view(n, 1, -1)      #(n, 1, h)
        target_onehot = torch.zeros(output.shape).cuda().scatter_(dim=1, index=target, value=1)  #(n, c, h)

        if weights is None:
            ce = -(F.log_softmax(output, dim=1) * target_onehot).sum(dim=1)     #(n, h)
            ce = ce.mean(dim=1)
            loss = ce.mean()
        else:
            weights = weights.float()
            weights = weights.contiguous()
            weights = weights.view(n, -1)   #(n, h)
            weights = weights / weights.sum(1, keepdim=True) * float(weights.size(1))

            ce = -(F.log_softmax(output, dim=1) * target_onehot).sum(dim=1)     #(n, h)
            ce = (ce * weights).mean(dim=1)
            loss = ce.mean()

        return loss


    @staticmethod
    def JaccardLoss(output, target, weights=None):
        # type:(Tensor, Tensor, Tensor) -> Tensor
        """
        Measures the criterion for classification.
        @param output: (N, C, H, W), float
        @param target: (N, 1, H, W), long
        @return loss: scalar
        """
        n = output.size(0)
        c = output.size(1)

        output = output.contiguous()
        output = output.view(n, c, -1)
        if c == 2:
            output = F.softmax(output, 1).float()[:, 1:, :]

        target = target.float()
        target = target.contiguous()
        target = target.view(n, 1, -1)
        # target_onehot = torch.zeros(output.shape).scatter_(dim=1, index=target, src=1)

        axis = [1, 2]
        inter = torch.sum(output * target, axis)        #(n, )
        union = torch.sum(output + target, axis) - inter

        iou = (inter + _epsilon) / (union + _epsilon)  # (n, )
        iou = iou.mean()
        loss = 1-iou

        return loss



    @staticmethod
    def JaccardCrossEntropyLoss(output, target, weights=None):
        # type:(Tensor, Tensor, Tensor) -> Tensor
        """
        Measures the criterion for classification.
        @param output: (N, C, H, W), float
        @param target: (N, 1, H, W), long
        @return loss: scalar
        """
        CE = F.cross_entropy(output, target)

        iou = 1.-Loss.JaccardLoss(output, target)
        iou = torch.clamp(iou, _epsilon, 1.)

        loss = -torch.log(iou).mean() + CE
        return loss


    @staticmethod
    def JaccardBCELossClass(output, target, weights=None):
        # type:(Tensor, Tensor, Tensor) -> Tensor
        """
        Measures the criterion for classification.
        @param output: (N, 1)
        @param target: (N, 1)
        @return loss: scalar
        """
        assert output.shape==target.shape, "input shape must be the same as target shape"
        assert len(output.shape) == 2
        assert torch.all((output>=0.) & (output <=1.)), "input value must be in [0., 1.]"
        assert torch.all((target>=0.) & (target<=1.)), "target value must be in [0., 1.]"

        output = output.float()
        target = target.float()

        axis = [0, 1]
        inter = torch.sum(output * target, axis)
        union = torch.sum(output + target, axis) - inter

        iou = (inter + _epsilon) / (union + _epsilon)  # (1, )
        iou = torch.clamp(iou, _epsilon, 1.)

        loss = -torch.log(iou).mean() + Loss.BCELoss(output, target)
        return loss

    @staticmethod
    def JaccardBCELoss(output, target, weights=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        """
        Measures the criterion -log(Jaccard Index) + BCELoss.
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
    def BCELoss(output, target, weights=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor
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
    output = torch.randint(0, 2, size=(5, 2)).float()
    target = torch.randint(0, 2, size=(5, )).long()
    res = Loss.JaccardCrossEntropyLossClass(output, target)
    print(res)

