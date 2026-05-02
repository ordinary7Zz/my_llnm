from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from new_code.utils import calculate_binary_metrics, calculate_binary_metrics_confidence_intervals


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device) -> tuple[dict[str, object], dict[str, list]]:
    model.eval()
    all_labels = []
    all_probs = []
    all_paths = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        rr = batch["rr"].to(device, non_blocking=True).float()
        demo = batch["demo"].to(device, non_blocking=True).float().view(-1, 1, 2)
        img_fea = batch["img_fea"].to(device, non_blocking=True).float().view(-1, 2, 1)
        sex = demo[:, :, 1].view(-1, 1, 1)
        age = demo[:, :, 0].view(-1, 1, 1)

        logits = model(images, rr, img_fea, sex, age)[0]
        pos_probs = torch.sigmoid(logits).view(-1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(pos_probs.cpu().numpy().tolist())
        all_paths.extend(batch["path"])

    metrics = calculate_binary_metrics(all_labels, all_probs)
    metrics["confidence_intervals"] = calculate_binary_metrics_confidence_intervals(all_labels, all_probs)
    outputs = {
        "paths": all_paths,
        "labels": all_labels,
        "probs": all_probs,
    }
    return metrics, outputs


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    loss_fn,
) -> tuple[float, dict[str, float]]:
    model.train()
    all_labels = []
    all_probs = []
    running_loss = 0.0
    sample_count = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        rr = batch["rr"].to(device, non_blocking=True).float()
        demo = batch["demo"].to(device, non_blocking=True).float().view(-1, 1, 2)
        img_fea = batch["img_fea"].to(device, non_blocking=True).float().view(-1, 2, 1)
        sex = demo[:, :, 1].view(-1, 1, 1)
        age = demo[:, :, 0].view(-1, 1, 1)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images, rr, img_fea, sex, age)[0]
            targets = labels.float().view(-1, 1)
            loss = loss_fn(logits.view(-1, 1), targets)
            pos_probs = torch.sigmoid(logits).view(-1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(pos_probs.detach().cpu().numpy().tolist())

    avg_loss = running_loss / max(sample_count, 1)
    metrics = calculate_binary_metrics(all_labels, all_probs)
    return avg_loss, metrics
