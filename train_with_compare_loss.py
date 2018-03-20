from network.models import MoreSim
from network.utils import new_hinge_loss, accuracy, get_confuse_rate
from network.utils import KITTI_3_Dataset
from network.logger import Logger

import torch
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import cv2


# torch.set_printoptions(precision=6)
gpus = [3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])

# Set to False if patch format is Lab.
patch_is_BGR = True
norm = True


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


def cast_to_range255(x):
    min = np.min(x)
    max = np.max(x)
    m = 255/(max - min)
    normal_x = (x-min) * m
    return normal_x.astype(int)


def to_RGB(x):
    # Color channel move to the last dim.
    x = x.permute(1, 2, 0).numpy()
    if not patch_is_BGR:
        x_converted = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
    else:
        # BGR to RGB. Visual effect is not good probably because of mean extraction.
        x_converted = x[:, :, [2, 0, 1]]
        if norm:
            x_converted = cast_to_range255(x_converted)
    return x_converted


def main():
    log_dir = "./log_with_compare_loss/compare_RGB_100pixel/"
    two_set_vars = False
    patchsize = 56
    out_features = 256
    image_channels = 3
    max_epochs = 50000

    judge = MoreSim(image_channels, out_features, two_set_vars=two_set_vars)

    # Use multiple GPUs
    if len(gpus) > 1:
        judge = DataParallel(judge.cuda(gpus[0]), device_ids=gpus)
    else:
        judge = judge.cuda()

    train_logger = Logger(log_dir=log_dir + "train")
    test_logger = Logger(log_dir=log_dir + "test")
    optimizer = optim.Adam(judge.parameters(), lr=0.0001)

    train_images = 160
    test_images  = 40
    train_patch_set = KITTI_3_Dataset(patchsize, cntImages=train_images, offset=0)
    train_patch_set.newData()
    train_loader = DataLoader(train_patch_set, batch_size=256, num_workers=4, pin_memory=True, drop_last=True)

    test_patch_set = KITTI_3_Dataset(patchsize, cntImages=test_images, offset=train_images)
    test_patch_set.newData()
    test_loader = DataLoader(test_patch_set, batch_size=256, num_workers=4, pin_memory=True, drop_last=True)

    test_confuse_rate = AverageMeter()

    for e in range(max_epochs):
        train_patch_set.newData()
        for i, pairs_d in enumerate(train_loader):
            step = e * len(train_loader) + i
            if len(gpus) == 1:
                pairs = Variable(pairs_d.cuda(async=True), requires_grad=False)
            else:
                pairs = Variable(pairs_d.cuda(gpus[0], async=True), requires_grad=False)

            loss, preds = judge(pairs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                train_confuse_rate = get_confuse_rate(preds)

                train_logger.log_scalar("confuse_rate", train_confuse_rate, step)
                train_logger.log_scalar("loss", to_np(loss), step)
                # train_logger.log_histogram("preds", to_np(preds), step)

                print("Step %d \t loss is %f" % (step, to_np(loss)))
                print("Pred_pos range: ", get_range(preds[0]))
                print("Pred_neg range: ", get_range(preds[1]))

                for tag, value in judge.named_parameters():
                    tag = tag.replace('.', '/')
                    train_logger.log_histogram(tag, to_np(value), step)
                    train_logger.log_histogram(tag+'/grad', to_np(value.grad), step)

                # Randomly select 1 (pos pair, neg pair) to log for visualization.
                pos_image_pairs = np.random.choice(np.arange(0, train_loader.batch_size), 1, replace=False)
                for idx in pos_image_pairs:
                    train_logger.log_images("pos", [to_RGB(pairs_d[idx][0]), to_RGB(pairs_d[idx][1])], step)
                    # train_logger.log_images("pos_1", to_RGB(pairs_d[idx][1]), step)
                    train_logger.log_images("neg", [to_RGB(pairs_d[idx][0]), to_RGB(pairs_d[idx][2])], step)

                if step % 100 == 0:
                    test_patch_set.newData()

                    for p in test_loader:
                        p = Variable(p.cuda(), requires_grad=False)
                        _, preds_test = judge(p)
                        test_confuse_rate.update(get_confuse_rate(preds_test))
                    test_logger.log_scalar("test_confuse_rate", test_confuse_rate.avg, step)
                    test_confuse_rate.reset()

                    if (step+1) % 10000 == 0:
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