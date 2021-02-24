import os
import sys
import cv2
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TrainValDataset
from model import RESCAN
from model import VGG
from cal_ssim import SSIM

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        self.net = RESCAN().cuda()
        self.l2 = MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.vgg=VGG().cuda()

        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        self.opt = Adam(self.net.parameters(), lr=settings.lr)
        self.sche = MultiStepLR(self.opt, milestones=[240000, 320000], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 1. torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
        print(model)
        print("The number of parameters: {}".format(num_params))
    def inf_batch(self, name, batch):
        if name == 'train':
            self.net.zero_grad()
        if self.step==0:
            self.print_network(self.net)
            
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        R=O-B
        rain_list, img = self.net(O)   # 对应网络的输出
        loss_list = [self.l2(rain,R) for rain in rain_list]
        ssim_list = [self.ssim(O-rain,B) for rain in rain_list]
        ssim_img = self.ssim(img,B)
        ssim_list.append(ssim_img)
        vgg_gt = self.vgg.forward(B)
        eval = self.vgg.forward(img)
        loss_vgg = [self.l1(eval[m], vgg_gt[m]) for m in range(len(vgg_gt))]
        loss_img_l1=self.l1(img,B)
        loss_rain = sum(loss_list)
        loss_img=sum(loss_vgg)+loss_img_l1
        loss= loss_rain+loss_img
        if name == 'train':
            loss.backward()
            self.opt.step()

        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)
        self.write(name, losses)

        return img

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train')
    #sess.tensorboard('val')

    dt_train = sess.get_dataloader('train')    
    #dt_val = sess.get_dataloader('val')

    while sess.step < 400000:
        sess.sche.step()
        sess.net.train()

        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train', batch_t)

        #if sess.step % 4 == 0:             
        #    sess.net.eval()
        #    try:
        #        batch_v = next(dt_val)
        #    except StopIteration:
        #        dt_val = sess.get_dataloader('val')
        #        batch_v = next(dt_val)
        #    pred_v = sess.inf_batch('val', batch_v)

        if sess.step % int(sess.save_steps / 16) == 0:     
            sess.save_checkpoints('latest')
        #if sess.step % int(sess.save_steps / 2) == 0:
        #    sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
        #    if sess.step % 4 == 0:
        #        sess.save_image('val', [batch_v['O'], pred_v, batch_v['B']])
        #        logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model)

