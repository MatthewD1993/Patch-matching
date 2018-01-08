from network.net_paper import Judge
from network.utils import accuracy, get_confuse_rate # , new_hinge_loss
from network.utils import KITTIPatchesDataset
from network.logger import Logger

import torch
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
# from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import cv2
import torch.nn.functional as F

def to_np(x):
    return x.data.cpu().numpy()


def get_range(x):
    if isinstance(x, Variable):
        x = x.data
    x = x.cpu().numpy()
    return np.min(x), np.max(x)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_RGB(x):
    x = x.permute(1, 2, 0).numpy()
    x_converted = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
    return x_converted

# torch.set_printoptions(precision=6)
gpus = [2]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])


def main():
    patchsize = 56
    features = 64
    out_features = features * 4
    image_channels = 3
    epochs = 10000

    # TODO Big control.
    # Random guss correct ratio
    # correct_ratio = 0.5
    # while correct_ratio < 0.8:  # Keep training

    judge = Judge(image_channels, out_features).cuda()
    # judge._initialize_weights()
    # judge = DataParallel(judge.cuda(), device_ids=gpus)

    train_logger = Logger("./log_t/train/")
    # test_logger = Logger("./log/test/")
    optimizer = optim.Adam(judge.parameters(), lr=0.0001)

    patch_set = KITTIPatchesDataset()
    patch_set.load_data("./data/test_patches.npy")
    # train_loader = DataLoader(patch_set, batch_size=64, num_workers=4, pin_memory=True, drop_last=True)

    # test_patch_set = KITTIPatchesDataset()
    # test_patch_set.load_data("./data/test_patches.npy")
    # test_loader = DataLoader(test_patch_set, batch_size=64, num_workers=4, pin_memory=True, drop_last=True)

    margin = 1.
    threshold = 0.3

    pairs_d = torch.FloatTensor(4,2,3,56,56)
    labels_d = torch.FloatTensor(4,1)
    pairs_d[0], labels_d[0] = patch_set[7]
    pairs_d[1], labels_d[1] = patch_set[6]
    pairs_d[2], labels_d[2] = patch_set[866]
    pairs_d[3], labels_d[3] = patch_set[867]


    for e in range(epochs):
        # patch_set.newData()
        step = e

        pairs = Variable(pairs_d.cuda(), requires_grad=False)
        # print("Input pairs shape(should be [Batch size,2,3,56,56])", pairs.shape)
        labels = Variable(labels_d.cuda(), requires_grad=False)

        # print("pairs shape", pairs.shape)

        preds = judge(pairs)

        loss = F.hinge_embedding_loss(preds, labels)

        # final_loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # test_acc = AverageMeter()
        # test_confuse_rate = AverageMeter()

        if step % 10 == 0:
            train_acc = accuracy(preds, labels, margin, threshold)
            train_confuse_rate = get_confuse_rate(preds)
            train_logger.log_scalar("accuracy", train_acc, step)
            train_logger.log_scalar("confuse_rate", train_confuse_rate, step)
            train_logger.log_histogram("preds", to_np(preds), step)

            print("Step %d \tloss is %f" % (step, to_np(loss)))
            print("Preds : ", to_np(preds))

            for tag, value in judge.named_parameters():
                tag = tag.replace('.', '/')
                train_logger.log_histogram(tag, to_np(value), step)
                train_logger.log_histogram(tag+'/grad', to_np(value.grad), step)

            pos_image_pairs = np.random.choice(np.arange(0, 2, 2), 1, replace=False)
            for idx in pos_image_pairs:
                train_logger.log_images("pos", [to_RGB(pairs_d[idx][0]), to_RGB(pairs_d[idx][1])], step)
                # train_logger.log_images("pos_1", to_RGB(pairs_d[idx][1]), step)
                train_logger.log_images("neg", [to_RGB(pairs_d[idx+1][0]), to_RGB(pairs_d[idx+1][1])], step)

            # neg_image_pairs = np.random.choice(np.arange(1, train_loader.batch_size, 2), 2, replace=False)
            # for idx in neg_image_pairs:
            #     train_logger.log_images("neg", [to_RGB(pairs_d[idx][0]), to_RGB(pairs_d[idx][1])], step)
            #     # train_logger.log_images("neg_1", to_RGB(pairs_d[idx][1]), step)

            if step % 100 == 0:
                train_logger.log_scalar("loss", to_np(loss), step)
                # print("Iteration {} loss: {}".format(step, to_np(loss)))

                # for p, l in test_loader:
                #     p = Variable(p.cuda(0,), requires_grad=False)
                #     l = Variable(l.cuda(0,), requires_grad=False)
                #     pred = judge(p)
                #     test_acc.update(accuracy(pred, l))
                #     test_confuse_rate.update(get_confuse_rate(pred))
                # test_logger.log_scalar("test_acc", test_acc.avg, step)
                # test_logger.log_scalar("test_confuse_rate", test_confuse_rate.avg, step)
                # test_acc.reset()

                # if i % 500 == 0:
                #     # Log train accuracy
                #     train_acc = accuracy(preds, labels)
                #     # Log test accuracy
                #     test_acc =  accuracy()

                if step % 1000 == 0:
                    torch.save(judge.state_dict(), "./log/check_"+str(step))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()