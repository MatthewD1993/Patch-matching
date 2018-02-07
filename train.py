from network.net_paper import Judge
from network.utils import new_hinge_loss, accuracy, get_confuse_rate
from network.utils import KITTIPatchesDataset
from network.logger import Logger

import torch
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
gpus = [0, 1, 2, 3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])


def main():
    log_dir = "./log_1set_vars/"
    two_set_vars = False
    patchsize = 56
    features = 64
    out_features = features * 4
    image_channels = 3
    epochs = 500000


    judge = Judge(image_channels, out_features, two_set_vars=two_set_vars)
    # judge._initialize_weights()
    judge = DataParallel(judge.cuda(gpus[0]), device_ids=gpus)

    train_logger = Logger(log_dir=log_dir + "train")
    test_logger = Logger(log_dir=log_dir + "test")
    optimizer = optim.Adam(judge.parameters(), lr=0.0001)

    patch_set = KITTIPatchesDataset(patchsize)
    patch_set.newData()
    train_loader = DataLoader(patch_set, batch_size=128, num_workers=4, pin_memory=True, drop_last=True)

    test_patch_set = KITTIPatchesDataset()
    test_patch_set.load_data("./data/test_patches.npy")
    test_loader = DataLoader(test_patch_set, batch_size=128, num_workers=4, pin_memory=True, drop_last=True)

    margin = 1.
    threshold = 0.3

    for e in range(epochs):
        patch_set.newData()
        for i, (pairs_d, labels_d) in enumerate(train_loader):
            step = e * len(train_loader) + i

            pairs = Variable(pairs_d.cuda(0, async=True), requires_grad=False)
            # print("Input pairs shape(should be [Batch size,2,3,56,56])", pairs.shape)
            labels = Variable(labels_d.cuda(0, async=True), requires_grad=False)

            preds = judge(pairs)

            loss = new_hinge_loss(preds, labels)

            # final_loss = torch.mean(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_acc = AverageMeter()
            test_confuse_rate = AverageMeter()

            if step % 40 == 0:
                train_acc = accuracy(preds, labels, margin, threshold)
                train_confuse_rate = get_confuse_rate(preds)
                train_logger.log_scalar("accuracy", train_acc, step)
                train_logger.log_scalar("confuse_rate", train_confuse_rate, step)
                train_logger.log_histogram("preds", to_np(preds), step)

                print("Step %d \tloss is %f" % (step, to_np(loss)))
                print("Preds range: ", get_range(preds))

                for tag, value in judge.named_parameters():
                    tag = tag.replace('.', '/')
                    train_logger.log_histogram(tag, to_np(value), step)
                    train_logger.log_histogram(tag+'/grad', to_np(value.grad), step)

                pos_image_pairs = np.random.choice(np.arange(0, train_loader.batch_size, 2), 1, replace=False)
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

                    for p, l in test_loader:
                        p = Variable(p.cuda(0,), requires_grad=False)
                        l = Variable(l.cuda(0,), requires_grad=False)
                        pred = judge(p)
                        test_acc.update(accuracy(pred, l))
                        test_confuse_rate.update(get_confuse_rate(pred))
                    test_logger.log_scalar("test_acc", test_acc.avg, step)
                    test_logger.log_scalar("test_confuse_rate", test_confuse_rate.avg, step)
                    test_acc.reset()

                    # if i % 500 == 0:
                    #     # Log train accuracy
                    #     train_acc = accuracy(preds, labels)
                    #     # Log test accuracy
                    #     test_acc =  accuracy()

                    if step % 20000 == 0:
                        torch.save(judge.state_dict(), log_dir+"check_"+str(step))


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