import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def eval_metrics(pred, gt, clip01=True):
    if clip01:
        pred = np.clip(pred, 0.0, 1.0)
    r2s   = [r2_score(gt[:, i], pred[:, i]) for i in range(3)]
    maes  = [mean_absolute_error(gt[:, i], pred[:, i]) for i in range(3)]
    rmses = [np.sqrt(mean_squared_error(gt[:, i], pred[:, i])) for i in range(3)]
    return r2s, maes, rmses

def interpret_confidence(prob):
    if prob > 0.9:
        return "Very High"
    elif prob > 0.7:
        return "High"
    return "Low"

def set_requires_grad(module, flag=True):
    for p in module.parameters():
        p.requires_grad = flag

def make_optimizer(model, stage, lr_config):
    params = []
    wd = lr_config["WEIGHT_DECAY"]
    
    if stage == "warmup":
        set_requires_grad(model.xrf_enc, True)
        set_requires_grad(model.xrf_pred, True)
        set_requires_grad(model.img_enc, False)
        set_requires_grad(model.img_delta, False)
        set_requires_grad(model.gate, False)

        params.append({"params": model.xrf_enc.parameters(), "lr": lr_config["LR_XRF"], "weight_decay": wd})
        params.append({"params": model.xrf_pred.parameters(), "lr": lr_config["LR_XRF"], "weight_decay": wd})

    elif stage == "residual":
        set_requires_grad(model.img_enc, True)
        set_requires_grad(model.img_delta, True)
        set_requires_grad(model.gate, True)
        set_requires_grad(model.xrf_enc, True)
        set_requires_grad(model.xrf_pred, False)

        params.append({"params": model.img_enc.parameters(), "lr": lr_config["LR_IMG"], "weight_decay": wd})
        params.append({"params": model.img_delta.parameters(), "lr": lr_config["LR_IMG"], "weight_decay": wd})
        params.append({"params": model.gate.parameters(), "lr": lr_config["LR_GATE"], "weight_decay": wd})
        params.append({"params": model.xrf_enc.parameters(), "lr": lr_config["LR_XRF"], "weight_decay": wd})

    else:  # finetune
        set_requires_grad(model.xrf_enc, True)
        set_requires_grad(model.xrf_pred, True)
        set_requires_grad(model.img_enc, True)
        set_requires_grad(model.img_delta, True)
        set_requires_grad(model.gate, True)

        params.append({"params": model.parameters(), "lr": lr_config["LR_ALL"], "weight_decay": wd})

    return optim.AdamW(params)
