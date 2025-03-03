import numpy as np
from tqdm import tqdm
import torch
from torch import nn



# 8. Training Functions using Smooth L1 Loss
def smooth_l1_loss(output, target, seq_lengths):
    mask = torch.zeros_like(target, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        mask[i, :length, :] = True
    loss = nn.SmoothL1Loss(reduction='none')(output, target)
    masked_loss = loss * mask.float()
    return masked_loss.sum() / mask.sum() if mask.sum() > 0 else 0



def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0
    batches = 0
    for features, targets, _, seq_lengths in tqdm(dataloader):
        if targets is None:
            continue
        optimizer.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features, seq_lengths)
        loss = smooth_l1_loss(outputs, targets, seq_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1
    return epoch_loss / batches if batches > 0 else float('inf')


def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    batches = 0
    with torch.no_grad():
        for features, targets, _, seq_lengths in tqdm(dataloader):
            if targets is None:
                continue
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features, seq_lengths)
            loss = smooth_l1_loss(outputs, targets, seq_lengths)
            val_loss += loss.item()
            batches += 1
    return val_loss / batches if batches > 0 else float('inf')


def calculate_tm_score(predicted, reference):
    l_ref = len(reference)
    if l_ref >= 30:
        d0 = 0.6 * (l_ref - 0.5) ** 0.5 - 2.5
    elif l_ref >= 24:
        d0 = 0.7
    elif l_ref >= 20:
        d0 = 0.6
    elif l_ref >= 16:
        d0 = 0.5
    elif l_ref >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    tm_score = 0
    for i in range(min(len(predicted), l_ref)):
        di = np.linalg.norm(predicted[i] - reference[i])
        tm_score += 1 / (1 + (di/d0)**2)
    return tm_score / l_ref


def evaluate_model(model, dataloader, device):
    model.eval()
    tm_scores = []
    with torch.no_grad():
        for features, targets, _, seq_lengths in dataloader:
            if targets is None:
                continue
            features = features.to(device)
            outputs = model(features, seq_lengths)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for i, length in enumerate(seq_lengths):
                pred_coords = outputs[i, :length, :]
                target_coords = targets[i, :length, :]
                tm_score = calculate_tm_score(pred_coords, target_coords)
                tm_scores.append(tm_score)
    return np.mean(tm_scores) if tm_scores else 0

