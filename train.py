"""
Author: Chengbiao Deng
ziyoububianmj@gmail.com
"""
from network.models import Judge, Judge_small, DilationJudge, DilationJudgeRGB
from network.utils import new_hinge_loss, accuracy, get_confuse_rate, update_lr
from network.utils import KITTIPatchesDataset, SintelPatchesDataset, ChairsPatchesDataset
from network.logger import Logger

import torch
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import cv2
from tqdm import tqdm


# Set to False if patch format is Lab.
patch_is_BGR = True


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
            'state_dict': net.module.state_dict(),
            'optimizer': optim.state_dict(),
        },
        ckpt_fname
    )


def main():
    # Configuration.
    gpus = [3,4]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])

    log_dir = "/cdengdata/patchmatching/sintel_f128_lr_5e-5/"

    saved_model = '/cdengdata/patchmatching/sintel_f128_lr_5e-5/check_epoch1200'
    dataset = 'Sintel'
    resume = True
    train = True
    two_set_vars = False
    patchsize = 56
    max_epochs = 50000
    start_epoch = 1201 if resume else 0
    lr = 2e-5
    # max_epochs = 60

    # judge = Judge_small(two_set_vars=two_set_vars)
    judge = Judge(two_set_vars=two_set_vars)

    # Use multiple GPUs
    if len(gpus) > 0:
        judge = DataParallel(judge.cuda(), device_ids=list(range(len(gpus))))

    optimizer = optim.Adam(judge.parameters(), lr=lr)

    if resume:
        print('>>> Restore from checkpoint: ', saved_model)
        ckpt = torch.load(saved_model)
        if train:
            judge.module.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            judge.module.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train_logger = Logger(log_dir=log_dir + "train")
    test_logger = Logger(log_dir=log_dir + "test")

    if dataset == 'KITTI':
        train_images = 160
        test_images = 40
        train_patch_set = KITTIPatchesDataset(patchsize, cntImages=train_images, offset=0)
        test_patch_set = KITTIPatchesDataset(patchsize, cntImages=test_images, offset=train_images)
    elif dataset == 'Sintel':
        train_images = 833  # 160
        test_images  = 208
        train_patch_set = SintelPatchesDataset(patchsize, cntImages=train_images, offset=0)
        test_patch_set  = SintelPatchesDataset(patchsize, cntImages=test_images, offset=train_images)
    elif dataset == 'Chairs':
        train_images = 4000  # 160
        test_images = 1000
        train_patch_set = ChairsPatchesDataset(patchsize, cntImages=train_images, offset=0)
        test_patch_set  = ChairsPatchesDataset(patchsize, cntImages=test_images, offset=train_images)

    train_patch_set.newData()
    train_loader = DataLoader(train_patch_set, batch_size=256, num_workers=4, pin_memory=True, drop_last=True)

    test_patch_set.newData()
    test_loader = DataLoader(test_patch_set, batch_size=128, num_workers=4, pin_memory=True, drop_last=True)

    margin = 1.
    threshold = 0.3

    test_acc = AverageMeter()
    test_confuse_rate = AverageMeter()

    for e in tqdm(range(start_epoch, max_epochs), ncols=100):

        train_patch_set.newData()
        epoch_batches = len(train_loader)
        progress = tqdm(enumerate(train_loader), total=epoch_batches, ncols=100, desc='Training epoch '+str(e))
        for i, (pairs_d, labels_d) in progress:
            step = e * epoch_batches + i

            # pairs = Variable(pairs_d.cuda(0, async=True), requires_grad=False)
            # # print("Input pairs shape(should be [Batch size,2,3,56,56])", pairs.shape)
            # labels = Variable(labels_d.cuda(0, async=True), requires_grad=False)

            pairs  = Variable(pairs_d.cuda(), requires_grad=False)
            labels = Variable(labels_d.cuda(), requires_grad=False)
            preds = judge(pairs)
            loss  = new_hinge_loss(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr = update_lr(optimizer, step)

            if step % 50 == 0:
                train_acc = accuracy(preds, labels, margin, threshold)
                train_confuse_rate = get_confuse_rate(preds)

                train_logger.log_scalar("accuracy", train_acc, step)
                train_logger.log_scalar("confuse_rate", train_confuse_rate, step)
                train_logger.log_scalar("loss", to_np(loss)[0], step)
                train_logger.log_histogram("preds", to_np(preds), step)
                train_logger.log_scalar("lr", lr, step)
                progress.set_description("Training epoch " + str(e) + " Preds range: " + str(get_range(preds)))
                # print("Step %d \t loss is %f" % (step, to_np(loss)))

                for tag, value in judge.named_parameters():
                    tag = tag.replace('.', '/')
                    train_logger.log_histogram(tag, to_np(value), step)
                    if train:
                        train_logger.log_histogram(tag+'/grad', to_np(value.grad), step)

                # Randomly select 1 (pos pair, neg pair) to log for visualization.
                # pos_image_pairs = np.random.choice(np.arange(0, train_loader.batch_size, 2), 1, replace=False)
                # for idx in pos_image_pairs:
                #     train_logger.log_images("pos", [to_RGB(pairs_d[idx][0]), to_RGB(pairs_d[idx][1])], step)
                #     # train_logger.log_images("pos_1", to_RGB(pairs_d[idx][1]), step)
                #     train_logger.log_images("neg", [to_RGB(pairs_d[idx+1][0]), to_RGB(pairs_d[idx+1][1])], step)

                if step % 100 == 0:
                    for p, l in test_loader:

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

        progress.close()
        if e % 10 == 0:
            print("Saving at epoch {} {}".format(e, log_dir))
            save_model(judge, optimizer, e, log_dir+"check_epoch"+str(e))


if __name__ == '__main__':
    main()
