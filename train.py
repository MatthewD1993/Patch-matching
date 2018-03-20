from network.models import Judge
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


# torch.set_printoptions(precision=6)
gpus = [2]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])

# Set to False if patch format is Lab.
patch_is_BGR = True


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
    # Color channel move to the last dim.
    x = x.permute(1, 2, 0).numpy()
    if not patch_is_BGR:
        x_converted = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
    else:
        x_converted = x[:, :, [2, 0, 1]]
    return x_converted


def save_model(net, optim, epoch, ckpt_fname):
    # state_dict = net.state_dict()
    # for key in state_dict.keys():
    #     state_dict[key] = state_dict[key].cpu()
    torch.save(
        {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optim.state_dict(),
        },
        ckpt_fname
    )


def main():
    # Configuration.
    log_dir = "./log_evaluate_chrisitan/"
    saved_model = './torch2pytorch/christian_weight.ckpt'
    resume = True
    train = False
    two_set_vars = False
    patchsize = 56
    max_epochs = 50000
    start_epoch = 0
    lr = 0.00005
    # max_epochs = 60

    judge = Judge(two_set_vars=two_set_vars)

    # Use multiple GPUs
    if len(gpus) > 1:
        judge = DataParallel(judge.cuda(gpus[0]), device_ids=gpus)
    else:
        judge = judge.cuda()
    optimizer = optim.Adam(judge.parameters(), lr=lr)

    if resume:
        ckpt = torch.load(saved_model)
        if train:
            judge.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            judge.load_state_dict(ckpt)

    train_logger = Logger(log_dir=log_dir + "train")
    test_logger = Logger(log_dir=log_dir + "test")

    train_images = 160
    test_images  = 40
    train_patch_set = KITTIPatchesDataset(patchsize, cntImages=train_images, offset=0)
    train_patch_set.newData()
    train_loader = DataLoader(train_patch_set, batch_size=256, num_workers=4, pin_memory=True, drop_last=True)

    test_patch_set = KITTIPatchesDataset(patchsize, cntImages=test_images, offset=train_images)
    test_patch_set.newData()
    test_loader = DataLoader(test_patch_set, batch_size=256, num_workers=4, pin_memory=True, drop_last=True)

    margin = 1.
    threshold = 0.3

    test_acc = AverageMeter()
    test_confuse_rate = AverageMeter()

    for e in range(start_epoch, max_epochs):
        train_patch_set.newData()
        for i, (pairs_d, labels_d) in enumerate(train_loader):
            step = e * len(train_loader) + i

            # pairs = Variable(pairs_d.cuda(0, async=True), requires_grad=False)
            # # print("Input pairs shape(should be [Batch size,2,3,56,56])", pairs.shape)
            # labels = Variable(labels_d.cuda(0, async=True), requires_grad=False)

            pairs = Variable(pairs_d.cuda(), requires_grad=False)
            labels = Variable(labels_d.cuda(), requires_grad=False)

            preds = judge(pairs)
            loss = new_hinge_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                train_acc = accuracy(preds, labels, margin, threshold)
                train_confuse_rate = get_confuse_rate(preds)

                train_logger.log_scalar("accuracy", train_acc, step)
                train_logger.log_scalar("confuse_rate", train_confuse_rate, step)
                train_logger.log_scalar("loss", to_np(loss), step)
                train_logger.log_histogram("preds", to_np(preds), step)

                print("Step %d \t loss is %f" % (step, to_np(loss)))
                print("Preds range: ", get_range(preds))

                for tag, value in judge.named_parameters():
                    tag = tag.replace('.', '/')
                    train_logger.log_histogram(tag, to_np(value), step)
                    train_logger.log_histogram(tag+'/grad', to_np(value.grad), step)

                # Randomly select 1 (pos pair, neg pair) to log for visualization.
                pos_image_pairs = np.random.choice(np.arange(0, train_loader.batch_size, 2), 1, replace=False)
                for idx in pos_image_pairs:
                    train_logger.log_images("pos", [to_RGB(pairs_d[idx][0]), to_RGB(pairs_d[idx][1])], step)
                    # train_logger.log_images("pos_1", to_RGB(pairs_d[idx][1]), step)
                    train_logger.log_images("neg", [to_RGB(pairs_d[idx+1][0]), to_RGB(pairs_d[idx+1][1])], step)

                if step % 100 == 0:
                    for p, l in test_loader:
                        # p = Variable(p.cuda(0,), requires_grad=False)
                        # l = Variable(l.cuda(0,), requires_grad=False)
                        p = Variable(p.cuda(), requires_grad=False)
                        l = Variable(l.cuda(), requires_grad=False)
                        pred = judge(p)
                        test_acc.update(accuracy(pred, l))
                        test_confuse_rate.update(get_confuse_rate(pred))
                    test_logger.log_scalar("accuracy", test_acc.avg, step)
                    test_logger.log_scalar("confuse_rate", test_confuse_rate.avg, step)

                    # Must reset after every test.
                    test_acc.reset()
                    test_confuse_rate.reset()

        if e % 10 == 0:
            save_model(judge, optimizer, e, log_dir+"check_epoch"+str(e))


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