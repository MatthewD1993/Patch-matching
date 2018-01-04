import torch
import patchselect as ps
from torch.autograd import Function, Variable
from torch.autograd.function import once_differentiable
from torch.utils.data import Dataset
import numpy as np
# import cv2


class KITTIPatchesDataset(Dataset):
    img1 = "/cdengdata/data_scene_flow/training/image_2/%6_10.png"
    img2 = "/cdengdata/data_scene_flow/training/image_2/%6_11.png"
    flow = "/cdengdata/data_scene_flow/training/flow_noc/%6_10.png"
    cntImages = 200

    def __init__(self, patchsize=None, scale=1, offset=0):
        if patchsize is not None:
            self.patch_selector = ps.init(self.img1, self.img2, self.flow, self.cntImages, patchsize, scale, offset)
            self.patch_size = patchsize
        self.data = None
        # self.newData(self.patch_selector, 10000)

    def newData(self, num_sample_pairs=10000):
        # 10000*2*2*patchsize*patchsiz*3 float numbers, about 1.47GB
        data = ps.newData(self.patch_selector, num_sample_pairs)

        self.data = torch.FloatTensor(data).view(num_sample_pairs*2, 2, self.patch_size, self.patch_size, -1)
        self.data = self.data.permute(0, 1, 4, 2, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # return self.data[index]
        return self.data[index], torch.FloatTensor([(index & 1)*(-2.) + 1.])

    def save_data(self, path):
        np.save(path, self.data.numpy())
        print("Data saved at: ", path)

    def load_data(self, path):
        self.data = torch.FloatTensor(np.load(path))


# def new_hinge_loss(pred, label, t, margin):
#     """
#     label 1 means match, 0 means not match.
#     :param pred:
#     :param label:
#     :param t:
#     :param margin:
#     :return:
#     """
#     # t = torch.FloatTensor([t]).cuda()
#     if torch.eq(label, 1).any():
#         # t1 = pred.sub(t)
#         loss = torch.clamp(pred-t, min=0)
#     else:
#         # t2 = margin.add(t).sub(pred)
#         loss = torch.clamp(margin+t-pred, min=0)
#
#     return loss

def new_hinge_loss(input, target, margin=1.0, t=0.3, size_average=True):
    return HingeEmbeddingLoss.apply(input, target, margin, t, size_average)


# TODO: Make new hinge loss
class HingeEmbeddingLoss(Function):

    # def __init__(self, margin=1, t=0.3, size_average=True):
    #     super(HingeEmbeddingLoss, self).__init__()
    #     self.margin = margin
    #     self.size_average = size_average

    @staticmethod
    def forward(ctx, input, target, margin, t, size_average):
        ctx.margin = margin
        ctx.t = t
        ctx.size_average = size_average

        buffer = input.new()
        buffer.resize_as_(input).copy_(input)
        buffer[torch.eq(target, -1.)] = 0
        buffer.add_(-t)
        buffer[torch.lt(buffer, 0)] = 0
        output = buffer.sum()

        buffer.fill_(ctx.margin + ctx.t).add_(-1, input)
        buffer.clamp(min=0)
        buffer[torch.eq(target, 1)] = 0
        output += buffer.sum()

        if ctx.size_average:
            output = output / input.nelement()
        print("forward", output)
        ctx.save_for_backward(input, target)
        return input.new((output, ))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # print("grad_output", grad_output)
        input, target = ctx.saved_tensors
        grad_input = input.new().resize_as_(input).copy_(target)
        grad_input[torch.mul(torch.eq(target, -1), torch.gt(input, ctx.margin+ctx.t))] = 0
        grad_input[torch.mul(torch.eq(target, 1), torch.lt(input, ctx.t))] = 0

        if ctx.size_average:
            grad_input.mul_(1./input.nelement())

        grad_input.mul_(grad_output[0])

        return grad_input, None, None, None, None


    # def forward(self, input, target):
    #     buffer = input.new()
    #     buffer.resize_as_(input).copy_(input)
    #     buffer[torch.eq(target, -1.)] = 0
    #     output = buffer.sum()
    #
    #     buffer.fill_(self.margin).add_(-1, input)
    #     buffer.clamp_(min=0)
    #     buffer[torch.eq(target, 1.)] = 0
    #     output += buffer.sum()
    #
    #     if self.size_average:
    #         output = output / input.nelement()
    #
    #     self.save_for_backward(input, target)
    #     return input.new((output,))
    #
    # def backward(self, grad_output):
    #     input, target = self.saved_tensors
    #     grad_input = input.new().resize_as_(input).copy_(target)
    #     grad_input[torch.mul(torch.eq(target, -1), torch.gt(input, self.margin))] = 0
    #
    #     if self.size_average:
    #         grad_input.mul_(1. / input.nelement())
    #
    #     if grad_output[0] != 1:
    #         grad_input.mul_(grad_output[0])
    #
    # return grad_input, None

def accuracy(pred, label, margin=1.0, t=0.3):
    if isinstance(pred, Variable):
        pred = pred.data
    if isinstance(label, Variable):
        label = label.data

    acc = pred.clone()
    acc = acc.fill_(0)
    acc[torch.mul(torch.eq(label, 1), torch.lt(pred, 0.5*(margin + t)))] = 1
    acc[torch.mul(torch.eq(label, -1), torch.gt(pred, 0.5*(margin+t)))] = 1
    return torch.mean(acc)


def get_confuse_rate(pred, label=None):
    """Assume the order in a batch is well preserved.
    Then, pred [p, n, p, n, ...] if correct: pred[2*i] < pred[2*i + 1]
    """
    if isinstance(pred, Variable):
        preds = pred.data.cpu().numpy()
    assert type(preds) is np.ndarray
    # confuse_status = np.zeros(preds.size)
    confuse_status = [preds[2*i] >= preds[2*i+1] for i in range(preds.shape[0]//2)]
    return np.mean(confuse_status)

