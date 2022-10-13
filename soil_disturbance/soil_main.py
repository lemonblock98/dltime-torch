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
from dltime.models.InceptionTime import InceptionTime
from dltime.models.FCN import FCN
from dltime.models.conv_atten import TSTransformerEncoderConvClassifier
from transformers import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from data_process import handle_dataset_3dims


# outputs dir
OUTPUT_DIR = './soil_outputs/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CFG = TrainConfig()

now = datetime.datetime.now().strftime('%Y-%m-%d %H.%M')
model_name = 'inception-attn'
mode = 'all'
pretrain = False

LOGGER = get_logger(OUTPUT_DIR + model_name + '_' + now)
LOGGER.info(f'========= Training =========')

# data process
data_for_train = ['zwy', 'zwy2', 'zwy3', 'zwy4', 'j11', 'j11_2', 'j11_md', 'j11_527', 'yqcc', 'yqcc2', 'sky', 'syf', 'syf2', 'zyq', 'zyq2']
# data_for_train = ['zwy', 'zwy2', 'zwy3', 'zwy4']
# data_for_train = ['sky']
train_data = []
test_data = []
for data_name in data_for_train:
    train_data.extend(load_pkl(f'./pickle_data/{data_name}_train_{CFG.data_len}.pkl'))
    test_data.extend(load_pkl(f'./pickle_data/{data_name}_test_{CFG.data_len}.pkl'))

train_data = shuffle(train_data)
train_x, train_label = handle_dataset_3dims(train_data, mode=mode)
test_x, test_label = handle_dataset_3dims(test_data, mode=mode)
train_x = np.swapaxes(train_x, 2, 1)
test_x = np.swapaxes(test_x, 2, 1)
print('Train data size:', train_x.shape, train_label.shape)

train_dataset = Soil_Dataset(train_x, train_label, normalize=None, channel_first=True)
test_dataset = Soil_Dataset(test_x, test_label, normalize=None, channel_first=True)

feat_dim = train_x.shape[-1]
max_len = train_dataset.max_len
num_classes = 3

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

# 是否启用 weights & bias
if CFG.wandb:
    import wandb
    wandb.login()
    anony = None

    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(project='dltime', 
                    name=CFG.project,
                    config=class2dict(CFG),
                    group='soil_train',
                    job_type="train",
                    anonymous=anony)

# 模型定义
if model_name == 'inception-attn':
    model = TSInceptionSelfAttnEncoderClassifier(
            feat_dim=feat_dim, 
            max_len=max_len, 
            d_model=256, 
            num_heads=4,
            num_layers=4,
            dim_feedforward=256,
            num_classes=3).to(CFG.device)
if model_name == 'inception':
    model = InceptionTime(c_in=feat_dim, c_out=3).to(CFG.device)
if model_name == 'FCN':
    layers = [128, 256, 128]
    model = FCN(c_in=feat_dim, c_out=3, layers=layers).to(CFG.device)
    if pretrain:
        model.load_state_dict(torch.load('outputs/FCN_prune_layer_16_32_16.pth'))
if model_name == 'conv-attn':
    model = TSTransformerEncoderConvClassifier(
            feat_dim=feat_dim, 
            max_len=max_len, 
            d_model=512, 
            num_heads=4,
            num_layers=4,
            dim_feedforward=None,
            num_classes=3).to(CFG.device)

model.apply(weight_init)

# 优化器
optimizer_parameters = model.parameters()
optimizer = torch.optim.Adam(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
scheduler = get_scheduler(CFG, optimizer, num_train_steps)

# 损失函数
criterion = nn.CrossEntropyLoss(reduction="mean")
best_score = 0.

# 训练
for epoch in range(CFG.epochs):

    start_time = time.time()

    # train
    avg_loss, avg_acc = train_fn(CFG, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

    # eval
    avg_val_loss, predictions = valid_fn(CFG, test_loader, model, criterion, CFG.device)
    
    # scoring
    score = accuracy_score(test_label, predictions)
    f1 = f1_score(test_label, predictions, average='macro')

    elapsed = time.time() - start_time

    LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
    LOGGER.info(f'Epoch {epoch+1} - Acc: {score:.4f} - F1: {f1:.4f}')
    
    if CFG.wandb:
        wandb.log({"epoch": epoch+1, 
                "avg_train_loss": avg_loss, 
                "avg_train_acc": avg_acc,
                "avg_val_loss": avg_val_loss,
                "score": score})

    if best_score < score:
        best_score = score
        LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
        torch.save(model.state_dict(), f"./outputs/{model_name}_{mode}_{now}_16_32_16.pth")


torch.cuda.empty_cache()
gc.collect()
if CFG.wandb:
    wandb.finish()
