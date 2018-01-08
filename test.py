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

# s_data = test_patch_set.data.permute(0, 1, 3, 4, 2)
# data = s_data.contiguous().view(num_sample_pairs, 2, 2, patch_size, patch_size, 3).numpy()
#
# print("Check dataset...")
# rand_pairs = np.random.choice(num_sample_pairs, 50)
# # plt.ion()
# # f = plt.figure("compare")
# # f.suptitle("Compare")
# #
# for i in rand_pairs:
#     print("Show image pairs ", i)
#     pos_0 = cv2.cvtColor(data[i][0][0], cv2.COLOR_LAB2BGR)
#     pos_1 = cv2.cvtColor(data[i][0][1], cv2.COLOR_LAB2BGR)
#     neg_0 = cv2.cvtColor(data[i][1][0], cv2.COLOR_LAB2BGR)
#     neg_1 = cv2.cvtColor(data[i][1][1], cv2.COLOR_LAB2BGR)
#
#     cv2.imshow("pos_0", pos_0)
#     cv2.imshow("pos_1", pos_1)
#     cv2.imshow("neg_0", neg_0)
#     cv2.imshow("neg_1", neg_1)
#
#     cv2.moveWindow("pos_0", 100, 100)
#     cv2.moveWindow("pos_1", 500, 100)
#     cv2.moveWindow("neg_0", 100, 500)
#     cv2.moveWindow("neg_1", 500, 500)
#     cv2.waitKey(0)

    # ax1 = f.add_subplot(221)
    # ax1.imshow(pos_0)
    # ax2 = f.add_subplot(222)
    # ax2.imshow(pos_1)
    # ax3 = f.add_subplot(223)
    # ax3.imshow(neg_0)
    # ax4 = f.add_subplot(224)
    # ax4.imshow(neg_1)
    # plt.show()


    # axarr[0, 0].imshow(pos_0)
    # axarr[0, 1].imshow(pos_1)
    # axarr[1, 0].imshow(neg_0)
    # axarr[1, 1].imshow(neg_1)
    #
    # axarr[0, 0].set_title("pos_0")
    # axarr[0, 1].set_title("pos_1")
    # axarr[1, 0].set_title("neg_0")
    # axarr[1, 1].set_title("neg_1")

    # input("Press Enter to continue...")
#
# plt.ioff()

# test_patch_set = KITTIPatchesDataset()
# test_patch_set.load_data("./data/test_patches.npy")
# print(test_patch_set.data.shape)
#
# test_loader = DataLoader(test_patch_set, batch_size=16, num_workers=4, pin_memory=True)
#
#
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#
#
# def transform_to_image(x):
#     x = x.numpy()
#     # mat_array = cv2.cv.fromarray(x)
#     x = np.transpose(x, [1, 2, 0])
#     new_x = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
#     cv2.imshow("image", new_x)

#     cv2.waitKey(5000)
#
#     print("Range {} {}".format(np.min(new_x), np.max(new_x)))
#     # x = np.divide(x, 255.0)
#     # for i in range(x.shape[0]):
#     #     x[i, ...] = np.divide(x[i, ...] - np.min(x[i, ...]), np.max(x[i, ...]) - np.min(x[i, ...]))
#     return x
#
# # plt.ion()
# for i, (pairs, labels) in enumerate(test_loader):
#     print(pairs.shape)
#     # s = np.random.choice(test_loader.batch_size)
#     # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
#     transform_to_image(pairs[0][0])
#     # transform_to_image(pairs[s][1])

    # ax1.imshow(transform_to_image(pairs[s][0]), aspect='equal')
    # # ax1.axis('equal')
    # ax2.imshow(transform_to_image(pairs[s][1]), aspect='equal')
    # # ax2.axis('equal')
    # if i & 1 == 0:
    #     plt.suptitle("positive pair")
    # else:
    #     plt.suptitle("negative pair")
    # plt.show()