import torch
import torch.nn as nn
from network.models import MultiMaxPooling, UnwarpPrepare, UnwarpPool, MultiPoolPrepare, Judge
from imageio import imread
import numpy as np


class DenseFeatureExtractor(nn.Module):
    def __init__(self, imH=436, imW=1024):
        super(DenseFeatureExtractor, self).__init__()
        self.a = nn.Conv2d(3, 64, 5)
        self.imH = imH
        self.imW = imW

        self.dense_feature_extractor = nn.Sequential(
            MultiPoolPrepare(56, 56),

            nn.Conv2d(3, 64, 5),
            # nn.MaxPool2d(2, 2),
            MultiMaxPooling(2, 2, 2, 2),
            nn.Tanh(),
            nn.Conv2d(64, 80, 5),
            nn.Tanh(),
            nn.Conv2d(80, 160, 5),
            # nn.MaxPool2d(2, 2),
            MultiMaxPooling(2, 2, 2, 2),
            nn.Tanh(),
            nn.Conv2d(160, 256, 5),
            nn.Tanh(),
            nn.Conv2d(256, 512, 5),
            nn.Tanh(),
            nn.Conv2d(512, 512, 1),
            nn.Tanh(),
            nn.Conv2d(512, 128, 1),
            nn.Tanh(),

            UnwarpPrepare(),
            UnwarpPool(128, imH/(2*2), imW/(2*2), 2, 2),
            UnwarpPool(128, imH/2, imW/2, 2, 2),
        )

    def forward(self, inputs):
        out = self.dense_feature_extractor.forward(inputs).view(-1, self.imH, self.imW)
        return out


#
# temp_m = Judge()
# print('Something: ', temp_m.feature_extractor_f)

# m = DenseFeatureExtractor()
# data = torch.FloatTensor(1,3,16,16)
# print('a ', m.a)
# print('Something ***: ', m.features)
#
img_path = '/cdengdata/MPI-Sintel-complete/small/clean/alley_1/frame_0001.png'
img = imread(img_path)
img = np.transpose(img, [2, 0, 1])
# img = np.expand_dims(img, axis=0)
print('img shape:', img.shape)
data = torch.FloatTensor(img)

# shape = img.shape[2:]
# print(*shape)
# model = DenseFeatureExtractor(*shape)
#
# out = model(data)
# print('output shape: ', out.shape())

a1 = MultiPoolPrepare(56, 56)

test_data = torch.FloatTensor(3,512,512)
o1 = a1(test_data)
print(o1.shape)
a2 = nn.Conv2d(3, 64, 5, padding=2)
o2 =a2(o1)
print(o2.shape)
a3 = MultiMaxPooling(2, 2, 2, 2)
o3 = a3(o2)
print(o3.shape)