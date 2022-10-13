import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))


import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import wandb
import gc
import pandas as pd
from train_helper import train_fn, valid_fn
from config import TrainConfig
from utils import get_logger, get_scheduler, load_pkl, weight_init
from dltime.data.ts_datasets import Soil_Dataset
from dltime.models.inception_atten import TSInceptionSelfAttnEncoderClassifier
from dltime.models.conv_atten import TSTransformerEncoderConvClassifier
from dltime.models.InceptionTime import InceptionTime
from dltime.models.FCN import FCN
from transformers import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from data_process import handle_dataset_3dims


# outputs dir
OUTPUT_DIR = './soil_outputs/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CFG = TrainConfig()

now = datetime.datetime.now().strftime('%Y-%m-%d %H.%M')
model_name = 'FCN'
LOGGER = get_logger(OUTPUT_DIR + 'test' + now)
LOGGER.info(f'========= Testing =========')

# 模型定义
# model = TSInceptionSelfAttnEncoderClassifier(
#         feat_dim=5, 
#         max_len=64, 
#         d_model=512, 
#         num_heads=4,
#         num_layers=4,
#         dim_feedforward=None,
#         num_classes=3).to(CFG.device)
model = TSInceptionSelfAttnEncoderClassifier(
            feat_dim=5, 
            max_len=64, 
            d_model=256, 
            num_heads=4,
            num_layers=4,
            dim_feedforward=256,
            num_classes=3).to(CFG.device)
layers = [128, 256, 128]
# model = FCN(c_in=5, c_out=3, layers=layers).to(CFG.device)
# model = InceptionTime(c_in=5, c_out=3).to(CFG.device)
model.load_state_dict(torch.load('./outputs/inception-attn_all_2022-10-09 17.26_16_32_16.pth'))

# data process
# data_for_test = ['syf', 'syf2', 'yqcc', 'yqcc2', 'zwy5', 'j11', 'j11_2', 'j11_md', 'j11_527', 'j11_709', 'j11_717', 'sky', 'sky2', 'sky3']
# data_for_test = ['zwy5']
data_for_test = ['sky2', 'sky3']
# data_for_test = ['zwy', 'zwy2', 'zwy3', 'zwy4', 'zwy5', 'j11', 'j11_2', 'j11_md', 'j11_527', 'yqcc', 'yqcc2', 'sky', 'syf', 'sky2', 'sky3', 'zyq2']
data_len = 64
for data_name in data_for_test:
    LOGGER.info(f'test for dataset {data_name}')
    
    test_data = load_pkl(f'./pickle_data/{data_name}_train_{data_len}.pkl') + load_pkl(f'./pickle_data/{data_name}_test_{data_len}.pkl')
    test_x, test_label = handle_dataset_3dims(test_data, mode="all")
    test_x = np.swapaxes(test_x, 2, 1)
    print('Test data size:', test_x.shape, test_label.shape)

    test_dataset = Soil_Dataset(test_x, test_label, normalize=None, channel_first=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    avg_test_loss, predictions = valid_fn(CFG, test_loader, model, criterion, CFG.device)
    
    score3class = accuracy_score(test_label, predictions)
    score2class = accuracy_score(test_label==0, predictions==0)
    far_label1 = np.sum((test_label==1) & (predictions==0)) / np.sum(test_label==1)
    far_label2 = np.sum((test_label==2) & (predictions==0)) / np.sum(test_label==2)
    LOGGER.info(f'soil3class score: {score3class}')
    LOGGER.info(f'soil2class score: {score2class}')
    LOGGER.info(f'jump wrong to dig: {far_label1}')
    LOGGER.info(f'walk wrong to dig: {far_label2}')
    LOGGER.info(confusion_matrix(test_label, predictions))


torch.cuda.empty_cache()
gc.collect()
