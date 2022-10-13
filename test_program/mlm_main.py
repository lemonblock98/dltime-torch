import os
from re import L
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
print(os.path.abspath('..'))

import torch
import torch.nn as nn
import time
import wandb
import gc
import pandas as pd
from train_helper import mlm_train_fn, mlm_valid_fn
from config import TrainConfig
from utils import get_logger, get_optimizer_params, get_scheduler
from dltime.data.ts_datasets import MLM_UCR_UEADataset
from dltime.data.tsc_dataset_names import multivariate_equal_length
from dltime.models.ts_transformer import TSTransformerEncoder, TSTransformerEncoderMLM
from transformers import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


OUTPUT_DIR = './outputs/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CFG = TrainConfig()
LOGGER = get_logger("mlm_train")
LOGGER.info(f'========= Training =========')

multivariate_dataset = ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", \
    "CharacterTrajectories", "FaceDetection", "HandMovementDirection", "Heartbeat", "NATOPS", \
    "PEMS-SF", "SpokenArabicDigits"]

for dataset_name in multivariate_dataset[:1]:
    
    try:
        LOGGER.info(f"MLM Train for dataset {dataset_name}")

        train_dataset = MLM_UCR_UEADataset(dataset_name, split="train", pt_ratio=1)
        test_dataset = MLM_UCR_UEADataset(dataset_name, split="test", pt_ratio=1)
        print(train_dataset[0]['input'].size(), train_dataset[0]["padding_mask"].size())

        feat_dim = train_dataset[0]['input'].shape[-1]
        max_len = train_dataset.max_len

        LOGGER.info(f"Train for Dataset {dataset_name}, feat_dim: {feat_dim}, max_len: {max_len}")
        LOGGER.info(f"Train Size: {len(train_dataset)} Test Size: {len(test_dataset)}")
            
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    except:
        LOGGER.info(f"Load Dataset error!")
        continue

    if CFG.wandb:
        import wandb
        wandb.login()
        anony = None

        def class2dict(f):
            return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

        run = wandb.init(project='dltime', 
                        name=CFG.project + "_mlm",
                        config=class2dict(CFG),
                        group=dataset_name,
                        job_type="train",
                        anonymous=anony)
    
    model = TSTransformerEncoderMLM(
        feat_dim=feat_dim, 
        max_len=max_len,
        d_model=128, n_heads=2, num_layers=2, 
        dim_feedforward=512).to(CFG.device)


    optimizer_parameters = model.parameters()
    # optimizer_parameters = get_optimizer_params(model, CFG.encoder_lr, CFG.decoder_lr)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    criterion = nn.MSELoss(reduction="none")
    best_score = 1000.

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = mlm_train_fn(CFG, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss = mlm_valid_fn(CFG, test_loader, model, criterion, CFG.device)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        if CFG.wandb:
            wandb.log({"epoch": epoch+1, 
                    "avg_train_loss": avg_loss, 
                    "avg_val_loss": avg_val_loss})
        
        if best_score > avg_val_loss:
            best_score = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR+f"{dataset_name}_best.pth")

    torch.cuda.empty_cache()
    gc.collect()
    if CFG.wandb:
        wandb.finish()
