import torch
import math
import time
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


import wandb
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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def mlm_train_fn(cfg, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, item in enumerate(train_loader):
        for k, v in item.items():
            item[k] = v.to(device)

        batch_size = item['input'].size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            outputs = model(item['input'], 'train')
            y_preds = outputs[0]
        
        loss = criterion(y_preds, item['output'])
        loss = torch.masked_select(loss.view(-1, 1), item['lm_mask'].view(-1, 1) == 1).mean()
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        
        # acc = accuracy_score(labels.cpu().numpy(), y_preds.argmax(dim=-1).cpu().numpy())
        # acces.update(acc, batch_size)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_last_lr()[0]))
        if cfg.wandb:
            wandb.log({"loss": losses.val,
                       "lr": scheduler.get_last_lr()[0]})
    
    return losses.avg

def mlm_valid_fn(cfg, valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, item in enumerate(valid_loader):
        for k, v in item.items():
            item[k] = v.to(device)

        batch_size = item['input'].size(0)
        with torch.no_grad():
            outputs = model(item['input'], 'train')
            y_preds = outputs[0]
        
        loss = criterion(y_preds, item['output'])
        loss = torch.masked_select(loss.view(-1, 1), item['lm_mask'].view(-1, 1) == 1).mean()
        
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))

    return losses.avg

def train_fn(cfg, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses, acces = AverageMeter(), AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, item in enumerate(train_loader):
        for k, v in item.items():
            item[k] = v.to(device)
        # print(item['input'].size())
        labels = item['label']
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_preds = model(item['input'])
        
        # print(y_preds.size(), labels.size())
        loss = criterion(y_preds, labels)
        # loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        
        acc = accuracy_score(labels.cpu().numpy(), y_preds.argmax(dim=-1).cpu().numpy())
        acces.update(acc, batch_size)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Acc: {acc.val:.4f}({acc.avg:.4f})'
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          acc=acces,
                          grad_norm=grad_norm,
                          lr=scheduler.get_last_lr()[0]))
        if cfg.wandb:
            wandb.log({"loss": losses.val,
                       "acc": acces.val,
                       "lr": scheduler.get_last_lr()[0]})
    
    return losses.avg, acces.avg

def valid_fn(cfg, valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, item in enumerate(valid_loader):
        for k, v in item.items():
            item[k] = v.to(device)

        labels = item['label']
        batch_size = labels.size(0)
        # print(item['input'].size())
        with torch.no_grad():
            y_preds = model(item['input'])

        loss = criterion(y_preds, labels)
        # loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(torch.argmax(y_preds, dim=-1).cpu().numpy())
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions
