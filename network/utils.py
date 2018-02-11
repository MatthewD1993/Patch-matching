import torch
import patchselect as ps
from torch.autograd import Function, Variable
from torch.autograd.function import once_differentiable
from torch.utils.data import Dataset
import numpy as np
import cv2

print('Patch selecet:', ps.__file__)


class KITTIPatchesDataset(Dataset):
    img1 = "/cdengdata/data_scene_flow/training/image_2/%6_10.png"
    img2 = "/cdengdata/data_scene_flow/training/image_2/%6_11.png"
    flow = "/cdengdata/data_scene_flow/training/flow_noc/%6_10.png"
    one_fetch = 16384

    def __init__(self, patchsize, scale=1, offset=0, cntImages=200):

        self.patch_selector = ps.init(self.img1, self.img2, self.flow, cntImages, patchsize, scale, offset)
        self.patch_size = patchsize
        self.data = None
        # self.newData(self.patch_selector, 10000)

    def newData(self, num_sample_pairs=one_fetch):
        # 10000*2*2*patchsize*patchsiz*3 float numbers, about 1.47GB
        data = ps.newData(self.patch_selector, num_sample_pairs)
        self.data = torch.FloatTensor(data).view(num_sample_pairs*2, 2, 3, self.patch_size, self.patch_size)
        # self.data = self.data.permute(0, 1, 4, 2, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], torch.FloatTensor([(index & 1)*(-2.) + 1.])

    def save_data(self, path):
        np.save(path, self.data.numpy())
        print("Data saved at: ", path)

    def load_data(self, path):
        self.data = torch.FloatTensor(np.load(path))


class KITTI_3_Dataset(Dataset):
    img1 = "/cdengdata/data_scene_flow/training/image_2/%6_10.png"
    img2 = "/cdengdata/data_scene_flow/training/image_2/%6_11.png"
    flow = "/cdengdata/data_scene_flow/training/flow_noc/%6_10.png"
    one_fetch = 2 << 12

    def __init__(self, patchsize, offset=0, scale=1, cntImages=200):
        self.patch_selector = ps.init(self.img1, self.img2, self.flow, cntImages, patchsize, scale, offset)
        self.patch_size = patchsize
        self.data = None

    def newData(self, num_samples=one_fetch, visualize =False):
        data = ps.newData(self.patch_selector, num_samples)
        print('data shape:', data.shape)
        self.data = torch.FloatTensor(data).view(num_samples, 4, 3, self.patch_size, self.patch_size)
        # Dim 2: 0 ref; 1 pos; 2 ref; 3 neg
        # self.data = self.data[:, [0,1,3], :, :, :] # Too time consuming.
        # self.data = self.data.permute(0, 1, 4, 2, 3)
        samples = np.random.choice(num_samples,3)
        if visualize:
            for i in samples:
                s = data[i, 0, 0, ...]
                s = np.transpose(s, (1, 2, 0))
                s = cv2.cvtColor(s, cv2.COLOR_LAB2RGB)
                cv2.imshow('img', s)
                cv2.waitKey(0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


def compare_loss(input_p, input_n):
    return CompareLoss()(input_p, input_n)


class CompareLoss(Function):
    def forward(self, input_p, input_n, t=0.1, size_average=True):
        self.size_average = size_average

        buffer = input_p.new()
        buffer.resize_as_(input_p).copy_(input_p)
        buffer = buffer - input_n + t
        buffer[torch.lt(buffer, 0)] = 0

        output = buffer.sum()
        if self.size_average:
            output = output / input_p.nelement()

        self.delta = buffer

        return input_p.new((output,))

    def backward(self, grad_output):
        size_average = self.size_average
        delta = self.delta
        grad_input_p = delta.new().resize_as_(delta)
        grad_input_n = delta.new().resize_as_(delta)

        grad_input_p.fill_(1)
        grad_input_n.fill_(-1)

        grad_input_p[torch.eq(delta, 0)] = 0
        grad_input_n[torch.eq(delta, 0)] = 0

        if size_average:
            grad_input_p = grad_input_p / delta.nelement()
            grad_input_n = grad_input_n / delta.nelement()

        if grad_output[0] != 1:
            grad_input_p.mul_(grad_output[0])
            grad_input_n.mul_(grad_output[0])
        return grad_input_p, grad_input_n, None, None

    # @staticmethod
    # @once_differentiable
    # def backward(ctx, grad_output):
    #     size_average = ctx.size_average
    #     input_p, input_n, delta = ctx.saved_tensors
    #     grad_input_p = input_p.new().resize_as_(input_p)
    #     grad_input_n = input_n.new().resize_as_(input_n)
    #
    #     grad_input_p.fill_(1)
    #     grad_input_n.fill_(-1)
    #
    #     grad_input_p[torch.lt(delta, 0)] = 0
    #     grad_input_n[torch.lt(delta, 0)] = 0
    #
    #     if size_average:
    #         grad_input_p = grad_input_p / input_p.nelement()
    #         grad_input_n = grad_input_n / input_n.nelement()
    #
    #     if grad_output[0] != 1:
    #         grad_input_p.mul_(grad_output[0])
    #         grad_input_n.mul_(grad_output[0])
    #     return grad_input_p, grad_input_n, None, None


def new_hinge_loss(input, target, margin=1.0, t=0.3, size_average=True):
    return HingeEmbeddingLoss.apply(input, target, margin, t, size_average)


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
        buffer.clamp_(min=0)
        buffer[torch.eq(target, 1)] = 0
        output += buffer.sum()

        if ctx.size_average:
            output = output / input.nelement()
        ctx.save_for_backward(input, target)
        return input.new((output, ))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        input, target = ctx.saved_tensors
        grad_input = input.new().resize_as_(input).copy_(target)
        grad_input[torch.mul(torch.eq(target, -1), torch.gt(input, ctx.margin+ctx.t))] = 0
        grad_input[torch.mul(torch.eq(target, 1), torch.lt(input, ctx.t))] = 0

        if ctx.size_average:
            grad_input.mul_(1. / input.nelement())

        if grad_output[0] != 1:
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


def get_confuse_rate(preds, label=None):
    """Assume the order in a batch is well preserved.
    Then, pred [p, n, p, n, ...] if correct: pred[2*i] < pred[2*i + 1]
    """
    if isinstance(preds, tuple):
        preds_pos = preds[0].data.cpu().numpy()
        preds_neg = preds[1].data.cpu().numpy()
        confuse_status = np.greater_equal(preds_pos, preds_neg)*1

    if isinstance(preds, Variable):
        preds = preds.data.cpu().numpy()
        assert type(preds) is np.ndarray
        # confuse_status = np.zeros(preds.size)
        confuse_status = [preds[2*i] >= preds[2*i+1] for i in range(preds.shape[0]//2)]
    return np.mean(confuse_status)

