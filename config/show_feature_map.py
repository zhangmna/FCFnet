import os
import sys
import cv2
import argparse
import math
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader

import settings
from dataset import ShowDataset
from model_output_feature_map import RESCAN 
from cal_ssim import SSIM

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) 
class Session:
    def __init__(self):
        self.show_dir = settings.show_dir_feature
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir_feature)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = RESCAN().cuda()
        self.dataloaders = {}
        self.ssim=SSIM().cuda()
    def get_dataloader(self, dataset_name):
        dataset = ShowDataset(dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        with torch.no_grad():
            O_Rs,feature_1 = self.net(O)
        #loss_list = [self.crit(O_R, B) for O_R in O_Rs]
        #ssim_list = [self.ssim(O_R, B) for O_R in O_Rs]
        #psnr=PSNR(O_Rs[0].data.cpu().numpy()*255, B.data.cpu().numpy()*255)
        #print('psnr:%4f-------------ssim:%4f'%(psnr,ssim_list[0]))

        return O_Rs[-1],feature_1

    def save_image(self, No, imgs):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            img_file = os.path.join(self.show_dir, '%02d_%d.jpg' % (No, i))
            cv2.imwrite(img_file, img)
    def save_image_feature(self,name,imgs):
        for i, img in enumerate(imgs):#i 代表第几个卷积层
            img=img.cpu().data.numpy()
            img=img.squeeze(0)
            print(img.shape)
            for j in range(settings.feature_map_num): #j代表卷积层的第几个feature map
                print(img[j])
                a = (img[j]* 255)#.numpy()
                a = np.clip(a, 0, 255)
                img_file = os.path.join(self.show_dir, '%s_%d_%d.png' % (name, i+1,j+1))
                cv2.imwrite(img_file, a)


def run_show(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    sess.net.eval()

    dt = sess.get_dataloader('show_feature')

    for i, batch in enumerate(dt):
        logger.info(i)
        imgs,fusion= sess.inf_batch('test', batch)
        sess.save_image(i, imgs)
        sess.save_image_feature('fusion',fusion)
        #sess.save_image_feature('f_2',feature_2)
        #sess.save_image_feature('final',feature_final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    
    run_show(args.model)

