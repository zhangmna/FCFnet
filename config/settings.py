import os
import logging

block_num=12 #为偶数
connection_style = 'dense_connection'#symmetric_connection;no_connection;multi_short_skip_connection,'dense-connection'
mid_channel=16

num_dense_dilation=4


use_se=False
dilation=True
unit='my_unit'# '9-8traditional'51969,'9-8res'51969,'8-8dilation'46666,'8-8GRU'47418,'my_unit'46327

aug_data = False # Set as False for fair comparison
batch_size = 10
patch_size = 100
lr = 1e-3

data_dir = '/media/supercong/d277df79-f0d6-4f1d-979c-d79f956a61e5/congwang/dataset/rain100L'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 400

num_workers = 8
num_GPU = 1
device_id = 0

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


