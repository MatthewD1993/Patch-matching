import torch.nn as nn
import torch
from collections import OrderedDict

log_path = './log_1set_vars/check_0'


class FlowNet_Combined(nn.Module):

    def __init__(self, args=None, batchNorm=True, div_flow=20):
        super(FlowNet_Combined, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        # self.rgb_max = args.rgb_max
        # Can not use a sequential model because of the extra connection.
        # self.feature_extractor = nn.Sequential( OrderedDict([
        #     ('conv1', nn.Conv2d(3, 64, kernel_size=5, padding=2, bias=True)),
        #     ('pool1', nn.MaxPool2d(2, 2)),
        #     ('act1',  nn.Tanh()),
        #     ('conv2', nn.Conv2d(64, 80, kernel_size=5, padding=2, bias=True)),
        #     ('act2',  nn.Tanh()),
        #     ('conv3', nn.Conv2d(80, 160, kernel_size=5, padding=2, bias=True)),
        #     ('pool2', nn.MaxPool2d(2, 2)),
        #     ('act2',  nn.Tanh()),
        #     ('conv4', nn.Conv2d(160, 256, kernel_size=5, padding=2, bias=True)),
        #     ('act4',  nn.Tanh()),
        #     ('conv5', nn.Conv2d(256, 512, kernel_size=5, padding=2, bias=True)),
        #     ('act5',  nn.Tanh()),
        #     ('conv6', nn.Conv2d(512, 256, kernel_size=1)),
        #     ('act6',  nn.Tanh())
        # ]))

        self.f_conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2, bias=True)
        self.f_pool = nn.MaxPool2d(2, 2)
        self.f_act  = nn.Tanh()
        self.f_conv2 = nn.Conv2d(64, 80, kernel_size=5, padding=2, bias=True)
        self.f_conv3 = nn.Conv2d(80, 160, kernel_size=5, padding=2, bias=True)
        self.f_conv4 = nn.Conv2d(160, 256, kernel_size=5, padding=2, bias=True)
        self.f_conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2, bias=True)
        self.f_conv6 = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max

        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]

        conv1_a = self.f_act(self.f_pool(self.f_conv1(x1)))
        conv2_a = self.f_act(self.f_conv2(conv1_a))
        conv3_a = self.f_act(self.f_pool(self.f_conv3(conv2_a)))
        conv4_a = self.f_act(self.f_conv4(conv3_a))
        conv5_a = self.f_act(self.f_conv5(conv4_a))
        f_map_a = self.f_act(self.f_conv6(conv5_a))

        conv1_b = self.f_act(self.f_pool(self.f_conv1(x2)))
        conv2_b = self.f_act(self.f_conv2(conv1_b))
        conv3_b = self.f_act(self.f_pool(self.f_conv3(conv2_b)))
        conv4_b = self.f_act(self.f_conv4(conv3_b))
        conv5_b = self.f_act(self.f_conv5(conv4_b))
        f_map_b = self.f_act(self.f_conv6(conv5_b))


model = FlowNet_Combined()
# f = open('temp.ckpt', 'wb+')
# torch.save(model.state_dict(), f)
# # print(torch.load('temp.ckpt'))
for m in model.modules():
    print(m)

transfer_dict = {'module.feature_extractor_f.0.weight':'f_conv1.weight',
                 'module.feature_extractor_f.0.bias':'f_conv1.bias',
                 'module.feature_extractor_f.3.weight':'f_conv2.weight',
                 'module.feature_extractor_f.3.bias':'f_conv2.bias',
                 'module.feature_extractor_f.5.weight':'f_conv3.weight',
                 'module.feature_extractor_f.5.bias':'f_conv3.bias',
                 'module.feature_extractor_f.8.weight':'f_conv4.weight',
                 'module.feature_extractor_f.8.bias':'f_conv4.bias',
                 'module.feature_extractor_f.10.weight':'f_conv5.weight',
                 'module.feature_extractor_f.10.bias':'f_conv5.bias',
                 'module.feature_extractor_f.12.weight':'f_conv6.weight',
                 'module.feature_extractor_f.12.bias':'f_conv6.bias'}

# saved_weights = open()
weights_dict = torch.load(log_path)
# print(weights_dict.keys())
weights_dict = OrderedDict([(transfer_dict[k], v) for k,v in weights_dict.items()])
print(weights_dict.keys())
model.load_state_dict(weights_dict)
print('Successful!')