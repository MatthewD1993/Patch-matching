import matplotlib.pyplot as plt
from network.utils import KITTIPatchesDataset
from torch.utils.data import DataLoader
import numpy as np

import cv2

#
patch_size = 56
num_sample_pairs = 10000

test_patch_set = KITTIPatchesDataset(patchsize=patch_size)
for i in range(100):
    test_patch_set.newData(num_sample_pairs=num_sample_pairs)
    test_patch_set.save_data("./data/test_patches")

