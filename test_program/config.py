
from pickle import FALSE


class TrainConfig:
    wandb = False            # whether use wandb
    project = 'TSC-Transformer'
    num_workers=4
    apex=False
    print_freq=100
    scheduler='cosine'      # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0.02
    epochs=50
    encoder_lr=1e-5
    decoder_lr=1e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=16
    fc_dropout=0.2
    max_len=256
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    train=True
    device='cuda:0'
    d_model=128
    pretrained=True
    extract_path='Multivariate_ts' # 'Multivariate_ts' or 'Univeriate_ts'
    add_cls = False

