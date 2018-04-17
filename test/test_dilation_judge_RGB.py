import matplotlib.pyplot as plt
from network.models import DilationJudgeRGB
import imageio
import torch
import numpy as np
import os

gpus = [2, 3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])

path = '/cdengdata/patchmatching/inception_RGB_Chairs/check_epoch340'
img_name = '00153_img1.ppm'
image = imageio.imread(img_name)


model = DilationJudgeRGB()
model.load_state_dict(torch.load(path)['state_dict'])

model = model.cuda()

image_v = torch.autograd.Variable(torch.FloatTensor(np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0))).cuda()
out_img = model.feature_extractor_f(image_v).cpu()

out = np.transpose(np.array(out_img.squeeze().data), [1, 2, 0])
plt.imshow(out)
plt.show()
