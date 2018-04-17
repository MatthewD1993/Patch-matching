import torch.nn as nn
import math
import copy
from .utils import compare_loss
import torch
import torch.nn.functional as F

class Judge(nn.Module):

    # Normalize each vector, and calculate cosine similarity, 1 means exact same.
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels=3, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(Judge, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        self.feature_extractor_f = nn.Sequential(
            nn.Conv2d(image_channels, 64, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(64, 80, 5),
            nn.Tanh(),
            nn.Conv2d(80, 160, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(160, 256, 5),
            nn.Tanh(),
            nn.Conv2d(256, 512, 5),
            nn.Tanh(),
            nn.Conv2d(512, 512, 1),
            nn.Tanh(),
            nn.Conv2d(512, 128, 1),
            nn.Tanh(),
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        assert init_weight, "Only true option is supported."

        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)

    def forward(self, sample):
        # print("Sample shape is:", sample.size())
        if self.two_set_vars:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])
            p1_features = self.feature_extractor_s.forward(sample[:, 1])
        else:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])
            p1_features = self.feature_extractor_f.forward(sample[:, 1])
        p0_f = p0_features.clone()
        p1_f = p1_features.clone()
        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
        # print("patch feature shape is", p0_features.size())
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()


class Judge_small(nn.Module):
    # Normalize each vector, and calculate cosine similarity, 1 means exact same.
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels=3, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(Judge_small, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        self.feature_extractor_f = nn.Sequential(
            nn.Conv2d(image_channels, 64, 7, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 5),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        assert init_weight, "Only true option is supported."

        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)

    def forward(self, sample):
        # print("Sample shape is:", sample.size())
        if self.two_set_vars:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])
            p1_features = self.feature_extractor_s.forward(sample[:, 1])
        else:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])
            p1_features = self.feature_extractor_f.forward(sample[:, 1])
        p0_f = p0_features.clone()
        p1_f = p1_features.clone()
        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
        # print("patch feature shape is", p0_features.size())
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            # print("Module", m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()


class MoreSim(nn.Module):
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels=3, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(MoreSim, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        self.feature_extractor_f = nn.Sequential(
            nn.Conv2d(image_channels, 64, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(64, 80, 5),
            nn.Tanh(),
            nn.Conv2d(80, 160, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(160, 256, 5),
            nn.Tanh(),
            nn.Conv2d(256, 512, 5),
            nn.Tanh(),
            nn.Conv2d(512, 512, 1),
            nn.Tanh(),
            nn.Conv2d(512, 256, 1)
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        self.compare_loss = compare_loss

        assert init_weight, "Only true option is supported."
        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)

    def forward(self, sample):
        # print("Sample shape is:", sample.size())
        if self.two_set_vars:
            p_ref_features = self.feature_extractor_f.forward(sample[:, 0])
            p_pos_features = self.feature_extractor_s.forward(sample[:, 1])
            p_neg_features = self.feature_extractor_s.forward(sample[:, 2])
        else:
            p_ref_features = self.feature_extractor_f.forward(sample[:, 0])
            p_pos_features = self.feature_extractor_f.forward(sample[:, 1])
            p_neg_features = self.feature_extractor_f.forward(sample[:, 2])
        p_ref_features = p_ref_features.clone()
        p_pos_features = p_pos_features.clone()
        p_neg_features = p_neg_features.clone()

        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred_pos = self.sim(p_ref_features.squeeze_(), p_pos_features.squeeze_())
        pred_neg = self.sim(p_ref_features.squeeze_(), p_neg_features.squeeze_())
        # print("patch feature shape is", p0_features.size())

        loss = self.compare_loss(pred_pos, pred_neg)
        return loss, (pred_pos, pred_neg)

    def _initialize_weights(self):
        for m in self.modules():
            print("Module", m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MoreSim_Small(nn.Module):
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels=3, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(MoreSim_Small, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        self.feature_extractor_f = nn.Sequential(
            nn.Conv2d(image_channels, 64, 7, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 5),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        self.compare_loss = compare_loss

        assert init_weight, "Only true option is supported."
        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)

    def forward(self, sample):
        # print("Sample shape is:", sample.size())
        if self.two_set_vars:
            p_ref_features = self.feature_extractor_f.forward(sample[:, 0])
            p_pos_features = self.feature_extractor_s.forward(sample[:, 1])
            p_neg_features = self.feature_extractor_s.forward(sample[:, 2])
        else:
            p_ref_features = self.feature_extractor_f.forward(sample[:, 0])
            p_pos_features = self.feature_extractor_f.forward(sample[:, 1])
            p_neg_features = self.feature_extractor_f.forward(sample[:, 2])
        p_ref_features = p_ref_features.clone()
        p_pos_features = p_pos_features.clone()
        p_neg_features = p_neg_features.clone()

        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred_pos = self.sim(p_ref_features.squeeze_(), p_pos_features.squeeze_())
        pred_neg = self.sim(p_ref_features.squeeze_(), p_neg_features.squeeze_())
        # print("patch feature shape is", p0_features.size())

        loss = self.compare_loss(pred_pos, pred_neg)
        return loss, (pred_pos, pred_neg)

    def _initialize_weights(self):
        for m in self.modules():
            print("Module", m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            # nn.BatchNorm2d(n1x1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            # nn.BatchNorm2d(n3x3red),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            # nn.BatchNorm2d(n3x3),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            # nn.BatchNorm2d(n5x5red),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(n5x5red, n5x5, kernel_size=5, dilation=2, padding=4),
            # nn.BatchNorm2d(n5x5),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        return torch.cat([y1, y2, y3], 1)


class DilationJudge(nn.Module):
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels=3, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(DilationJudge, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        # self.receptive_field = 8 * 4 + 1
        self.pad_size = 8*2
        self.feature_extractor_f = nn.Sequential(
            Inception(image_channels, 32, 48, 48, 48, 48),
            Inception(128, 32, 48, 48, 48, 48),
            Inception(128, 32, 48, 48, 48, 48),
            Inception(128, 32, 48, 48, 48, 48)
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        assert init_weight, "Only true option is supported."

        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)

    def forward(self, sample):
        # Extract the center vector for comparison.
        if self.two_set_vars:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])[:, :, self.pad_size, self.pad_size]
            p1_features = self.feature_extractor_s.forward(sample[:, 1])[:, :, self.pad_size, self.pad_size]
        else:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])[:, :, self.pad_size, self.pad_size]
            p1_features = self.feature_extractor_f.forward(sample[:, 1])[:, :, self.pad_size, self.pad_size]
        p0_f = p0_features.clone()
        p1_f = p1_features.clone()
        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
        # print("patch feature shape is", p0_features.size())
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()


class DilationJudgeRGB(nn.Module):
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels=3, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(DilationJudgeRGB, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        # self.receptive_field = 8 * 4 + 1
        self.pad_size = 8*2
        self.feature_extractor_f = nn.Sequential(
            Inception(image_channels, 32, 48, 48, 48, 48),
            Inception(128, 32, 48, 48, 48, 48),
            Inception(128, 32, 48, 48, 48, 48),
            Inception(128, 32, 48, 48, 48, 48),
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32,  3,  1),
            nn.Sigmoid()
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        assert init_weight, "Only true option is supported."

        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)

    def forward(self, sample):
        # Extract the center vector for comparison.
        if self.two_set_vars:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])[:, :, self.pad_size, self.pad_size]
            p1_features = self.feature_extractor_s.forward(sample[:, 1])[:, :, self.pad_size, self.pad_size]
        else:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])[:, :, self.pad_size, self.pad_size]
            p1_features = self.feature_extractor_f.forward(sample[:, 1])[:, :, self.pad_size, self.pad_size]
        p0_f = p0_features.clone()
        p1_f = p1_features.clone()
        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
        # print("patch feature shape is", p0_features.size())
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()



class MultiPoolPrepare(nn.Module):
    def __init__(self, patch_y, patch_x):
        super(MultiPoolPrepare, self).__init__()
        # Pad params for image: (up, down, left, right)
        # If patch size is even, the left and right padding is not the same in order to keep same output size.
        self.pad_params = (patch_y // 2, (patch_y - 1) // 2, patch_x // 2, (patch_x - 1) // 2)

    def forward(self, inputs):
        padded = F.pad(inputs, self.pad_params)
        return padded.unsqueeze(0)


class UnwarpPrepare(nn.Module):
    def __init__(self):
        super(UnwarpPrepare, self).__init__()

    def forward(self, inputs):
        dim0 = inputs.size()[0]
        inputs.view(dim0, -1)
        return inputs.t()  # Transpose


class UnwarpPool(nn.Module):
    def __init__(self, outChans, curImgH, curImgW, dH, dW, ):
        super(UnwarpPool, self).__init__()
        self.outChans = outChans
        self.curImgH = curImgH
        self.curImgW = curImgW
        self.dH = dH
        self.dW = dW

    def forward(self, inputs):
        new_inputs = inputs.view(self.outChans, self.curImgH, self.curImgW, self.dH, self.dW, -1)
        return torch.transpose(new_inputs, 4, 5)


class MultiMaxPooling(nn.Module):
    def __init__(self, kH, kW, dH, dW):
        super(MultiMaxPooling, self).__init__()
        # self.kW = kW
        # self.kH = kH
        # self.dW = dW
        # self.dH = dH
        self.multi_pooling = nn.ModuleList()
        for i in range(dH):
            for j in range(dW):
                # Attention: Torch and Pytorch have opposite order for putting H and W.
                self.multi_pooling.append(nn.MaxPool2d( (kH, kW), stride=(dH, dW), padding=(-i, -j) ))

    def forward(self, inputs):
        print('num of pooling layers: ', len(self.multi_pooling))
        out = [pool(inputs) for pool in self.multi_pooling]
        for i in out:
            print("out ", i.shape)
        return torch.cat(out, dim=0)