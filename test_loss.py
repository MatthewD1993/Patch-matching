# # Test grad==========================================

# from network.utils import new_hinge_loss, accuracy
# import torch.nn.functional as F
# import torch
# from torch.autograd import Variable, gradcheck
#
# a = Variable(torch.FloatTensor([0.6, 1.7]), requires_grad=True)
# t = Variable(torch.FloatTensor([1, -1]))
#
# # loss = new_hinge_loss(a, t)
# loss = F.hinge_embedding_loss(a,t)
# print(loss.data)
# loss.backward()
# print(a.grad)
# print(t.grad)
#
# test = gradcheck(new_hinge_loss, (a,t), eps=1e-6, atol=1e-4)
# print(test)

# =================================
import matplotlib.pyplot as plt
import helper.flowlib as F
import helper.KITTI as K
import numpy as np
from network.utils import KITTIPatchesDataset
from torch.utils.data import DataLoader
import cv2

# def plot(pairs):
#     pairs = pairs.permute(0, 1, 3, 4, 2)
#     pairs = np.reshape(pairs.numpy(), [-1, 56, 56, 3])
#     for i in range(pairs.shape[0]):
#         print(pairs.shape[0]/4)
#         plt.subplot(int(pairs.shape[0]/4), 4, i+1)
#         temp = cv2.cvtColor(pairs[i], cv2.COLOR_LAB2RGB)
#         # m = np.array((56,56,3),np.uint8)
#
#         print(temp.shape)
#         print(temp.dtype)
#         # print(np.ptp(temp))
#         plt.imshow(temp)
#     # plt.show()
#     input("Press Enter to continue...")
#     return


def plot(pairs):
    """
    Plot one torch batch data.
    :param pairs: torch tensor [batch_size, 2, 3, 56, 56]
    :return:
    """
    pairs = pairs.permute(0, 1, 3, 4, 2)
    pairs = np.reshape(pairs.numpy(), [-1, 56, 56, 3])
    for i in range(pairs.shape[0]):
        print(pairs.shape[0] / 4)
        plt.subplot(int(pairs.shape[0] / 4), 4, i + 1)
        temp = cv2.cvtColor(pairs[i], cv2.COLOR_LAB2RGB)
        # m = np.array((56,56,3),np.uint8)

        print(temp.shape)
        print(temp.dtype)
        # print(np.ptp(temp))
        plt.imshow(temp)
    plt.draw()
    plt.pause(1)
    plt.waitforbuttonpress(0)
    return

test_patch_set = KITTIPatchesDataset()
test_patch_set.load_data("./data/test_patches.npy")
print(test_patch_set.data.shape)

test_loader = DataLoader(test_patch_set, batch_size=8, num_workers=4, pin_memory=True)
fig = plt.figure()
for i, (pairs_d, labels_d) in enumerate(test_loader):
    print(pairs_d.shape)
    # plt.ion()
    plot(pairs_d)


    # input("Press Enter to continue...")
    # cv2.waitKey(0)
    # plt.close('all')
    # plt.pause(0.0001)

    # plt.close()
