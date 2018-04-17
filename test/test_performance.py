from network.utils import SintelPatchesDataset
import cv2
import numpy as np

test_patch_set = SintelPatchesDataset(56, cntImages=900, offset=100)
for i in range(5):
    test_patch_set.newData(16384)
    seq_data = test_patch_set.data.contiguous().view(-1, 3, 56, 56)
    seq_data = seq_data.numpy()
    w_seq_data = np.transpose(seq_data, (0, 2, 3, 1))
    # for i in range(8):
    #     cv2.imshow('patch', w_seq_data[i, ...])
    #     cv2.waitKey(0)