from network.utils import new_hinge_loss, accuracy
import torch.nn.functional as F
import torch
from torch.autograd import Variable, gradcheck
#
from network.net_paper import Judge

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


def to_np(x):
    return x.data.cpu().numpy()


def plot(pairs, preds, labels):
    """
    Plot one torch batch data.
    :param pairs: torch tensor [batch_size, 2, 3, 56, 56]
    :return:
    """
    pairs = pairs.permute(0, 1, 3, 4, 2)
    pairs = np.reshape(pairs.numpy(), [-1, 56, 56, 3])
    for i in range(pairs.shape[0]):
        # print(pairs.shape[0] / 4)

        plt.subplot(int(pairs.shape[0] / 2), 2, i + 1)
        if i & 1 == 0:
            plt.title('prediction '+str(preds[i//2]) + 'label: ' + str(labels[i//2]))
        temp = cv2.cvtColor(pairs[i], cv2.COLOR_LAB2RGB)
        # m = np.array((56,56,3),np.uint8)
        plt.axis('off')

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
pretrained_weight = torch.load('./log/check_810000')
pretrained_weight = {k[7:]: v for k,v in pretrained_weight.items()}
judge = Judge(3, 256, two_set_vars=True)
print(judge)
# print('###############')
# print(pretrained_weight)
judge.load_state_dict(pretrained_weight)

test_loader = DataLoader(test_patch_set, batch_size=2, num_workers=4, pin_memory=True, shuffle=True)
fig = plt.figure()
for i, (pairs_d, labels_d) in enumerate(test_loader):
    print(pairs_d.shape)
    pairs_d_v = Variable(pairs_d)
    preds = judge(pairs_d_v)
    # plt.ion()
    plot(pairs_d, to_np(preds), labels_d.numpy())


    # input("Press Enter to continue...")
    # cv2.waitKey(0)
    # plt.close('all')
    # plt.pause(0.0001)

    # plt.close()