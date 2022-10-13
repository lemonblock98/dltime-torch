import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))


import torch
import torch.nn as nn
import time
import datetime
import wandb
import gc
import pandas as pd
from train_helper import train_fn, valid_fn
from config import TrainConfig
from utils import get_logger, get_optimizer_params, get_scheduler, load_pretrained_state_dict
from dltime.data.ts_datasets import UCR_UEADataset
from dltime.data.tsc_dataset_names import *
from dltime.models.inception_atten import TSInceptionSelfAttnEncoderClassifier
from transformers import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# outputs dir
OUTPUT_DIR = './outputs/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CFG = TrainConfig()

now = datetime.datetime.now().strftime('%Y-%m-%d %H.%M')
model_name = 'GTN'
LOGGER = get_logger(OUTPUT_DIR + now)
LOGGER.info(f'========= Training =========')

multivariate_dataset = ["JapaneseVowels", "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", \
    "CharacterTrajectories", "FaceDetection", "HandMovementDirection", "Heartbeat", "NATOPS", "SpokenArabicDigits"]

# univariate_dataset = ["CricketX", "ECG200", "Wafer"]

# for dataset_name in multivariate_dataset[:2]:
for dataset_name in multivariate_dataset:
    
    try:
        LOGGER.info(f"Train for dataset {dataset_name}")
        
        train_dataset = UCR_UEADataset(dataset_name, split="train", extract_path=CFG.extract_path, channel_first=True)
        test_dataset = UCR_UEADataset(dataset_name, split="test", extract_path=CFG.extract_path, channel_first=True)

        feat_dim = train_dataset[0]['input'].shape[-2]
        max_len = train_dataset.max_len
        num_classes = len(train_dataset.y2label)

        LOGGER.info(f"Train for Dataset {dataset_name}, feat_dim: {feat_dim}, max_len: {max_len}, num_class: {num_classes}")
        LOGGER.info(f"Train Size: {len(train_dataset)} Test Size: {len(test_dataset)}")
        LOGGER.info(f"Train Sample Size: {train_dataset[0]['input'].size()}")
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        test_labels = test_dataset.y
    except:
        LOGGER.info(f"Load Dataset error!")
        continue

    if CFG.wandb:
        # 启用 weights & bias
        import wandb
        wandb.login()
        anony = None

        def class2dict(f):
            return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

        run = wandb.init(project='dltime', 
                        name=CFG.project,
                        config=class2dict(CFG),
                        group=dataset_name + '_train',
                        job_type="train",
                        anonymous=anony)

    model = TSInceptionSelfAttnEncoderClassifier(
        feat_dim=feat_dim, 
        max_len=max_len, 
        d_model=512, 
        num_heads=4,
        num_layers=4,
        dim_feedforward=512,
        num_classes=num_classes).to(CFG.device)


    optimizer_parameters = model.parameters()
    # optimizer_parameters = get_optimizer_params(model, CFG.encoder_lr, CFG.decoder_lr)
    optimizer = torch.optim.Adam(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    best_score = 0.

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss, avg_acc = train_fn(CFG, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, predictions = valid_fn(CFG, test_loader, model, criterion, CFG.device)
        
        # scoring
        score = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='macro')

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
            torch.save(model.state_dict(), OUTPUT_DIR+f"{dataset_name}_{model_name}_best_retrain.pth")

    torch.cuda.empty_cache()
    gc.collect()
    if CFG.wandb:
        wandb.finish()
