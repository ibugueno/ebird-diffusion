#!/usr/bin/env python3

from __future__ import annotations
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ------------------- módulos del usuario ------------------- #
from UnetClass2 import Unet
from UnetClass2 import CombinedUnet  # asume composición completa (unet + partial_unet)
from Scheduler import LinearNoiseScheduler
from PairedDataSet_ajustable import PairedImageDataset

# ------------------- utilidades generales ------------------ #

def seed_everything(seed: int = 44):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Path, level=logging.INFO):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ("train_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logging to %s", log_file)


def save_checkpoint(state: dict, ckpt_path: Path, is_best: bool = False):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = ckpt_path.with_stem("Combinet" + ckpt_path.stem)
        torch.save(state, best_path)
        logging.info("Nuevo mejor modelo guardado en %s", best_path)


def load_checkpoint(model: nn.Module, optimizer: Adam, ckpt_path: Path):
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        logging.info("Checkpoint cargado (%s, epoch %d)", ckpt_path, ckpt["epoch"])
        return start_epoch, best_loss
    logging.warning("No se encontró checkpoint en %s", ckpt_path)
    return 0, float("inf")

# ------------------- bucles de entrenamiento ------------------- #

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, grad_clip=None):
    model.train()
    losses: List[float] = []
    for im, cond in tqdm(loader, desc="Train", leave=False):
        im = im.float().to(device, non_blocking=True)
        cond = cond.float().to(device, non_blocking=True)
        noise = torch.randn_like(im)
        timesteps = torch.randint(0, scheduler.num_timesteps, (im.shape[0],), device=device)
        noisy_im = scheduler.add_noise(im, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type):
            noise_pred = model(noisy_im, cond, timesteps)
            loss = criterion(noise_pred, noise)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
    return float(np.mean(losses))

# (Opcional) fase de validación

def validate(model, loader, criterion, scheduler, device):
    model.eval()
    val_losses: List[float] = []
    with torch.no_grad():
        for im, cond in tqdm(loader, desc="Val", leave=False):
            im = im.float().to(device, non_blocking=True)
            cond = cond.float().to(device, non_blocking=True)
            noise = torch.randn_like(im)
            timesteps = torch.randint(0, scheduler.num_timesteps, (im.shape[0],), device=device)
            noisy_im = scheduler.add_noise(im, noise, timesteps)
            noise_pred = model(noisy_im, cond, timesteps)
            loss = criterion(noise_pred, noise)
            val_losses.append(loss.item())
    return float(np.mean(val_losses))

# ------------------- función principal por subset ------------- #

def run_subset(cfg: dict, subset_ratio: float, resume: bool, device: torch.device, path_images: list, path_events: list, class_name: str):
    """Ejecuta entrenamiento para un subset específico."""
    diffusion_cfg = cfg["diffusion_params"]
    
    ###
    data_cfg = cfg["dataset_params"]
    path_images = path_images
    path_events = path_events
    ###
    
    model_cfg = cfg["model_params"]
    train_cfg = cfg["train_params"]
    save_name = train_cfg["ckpt_name"]

    # Directorio de salida
    run_dir = Path(train_cfg["task_name"]) /class_name /f"subset_{int(subset_ratio*100)}"
    setup_logger(run_dir)
    writer = SummaryWriter(log_dir=run_dir / "tb")

    # Dataset & DataLoader
    dataset = PairedImageDataset(
        path_images,
        path_events,
        im_size=tuple(data_cfg.get("im_size", (28, 28))),
        subset_ratio=subset_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # Modelo

    model = CombinedUnet(model_cfg, model_cfg).to(device)

    # Opción: cargar pesos base (e.g., unet pre‑entrenado)
    base_ckpt = Path(train_cfg["task_name"],"DDPM/checkpoints/ddpm_ckpt.pth")
    print("Base CKPT DDPM",base_ckpt)
    ckpt = torch.load(base_ckpt, map_location='cpu')
    if base_ckpt.is_file():
        try:
            model.unet.load_state_dict(ckpt['model_state_dict'])
            logging.info("Pesos base de Unet cargados desde %s", base_ckpt)
        except Exception as e:
            logging.warning("No se pudieron cargar pesos base: %s", e)

    scheduler_noise = LinearNoiseScheduler(
        num_timesteps=diffusion_cfg["num_timesteps"],
        beta_start=diffusion_cfg["beta_start"],
        beta_end=diffusion_cfg["beta_end"],
    )
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.MSELoss()

    # Checkpointing
    ckpt_path = run_dir / "checkpoints" / f"subset_{int(subset_ratio*100)}.pth"
    start_epoch, best_loss = (0, float("inf"))
    if resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, ckpt_path)

    # Entrenamiento
    num_epochs = train_cfg["num_epochs"]
    grad_clip = train_cfg.get("grad_clip", None)
    try:
        for epoch in range(start_epoch, num_epochs):
            logging.info("Epoch %d/%d (subset %.2f)", epoch + 1, num_epochs, subset_ratio)
            train_loss = train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler_noise, device, grad_clip)
            writer.add_scalar("Loss/train", train_loss, epoch)
            logging.info("Train loss: %.6f", train_loss)

            # Sin conjunto de validación – usa train para best
            val_loss = train_loss
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                ckpt_path,
                is_best=is_best,
            )
    except KeyboardInterrupt:
        logging.warning("Entrenamiento interrumpido. Guardando estado…")
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            ckpt_path,
        )
    finally:
        writer.close()
        logging.info("Fin del subset %.2f", subset_ratio)

# ------------------- CLI principal ------------------- #

def main(path_images: list, path_events: list, class_name: str):
    parser = argparse.ArgumentParser(description="Entrenamiento Combined‑UNet DDPM con subsets")
    parser.add_argument("--config", default="src/default.yaml", type=str, help="Ruta al YAML de configuración")
    parser.add_argument("--subset-ratios", nargs="*", default=[0.25, 0.5, 0.75, 1.0], type=float, help="Ratios de subconjunto a entrenar")
    parser.add_argument("--no-resume", action="store_true", help="No reanudar entrenamientos previos")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("train_params", {}).get("seed", 44))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    for ratio in args.subset_ratios:
        run_subset(cfg, subset_ratio=ratio, resume=not args.no_resume, device=device, path_images = path_images, path_events = path_events, class_name = class_name)

if __name__ == "__main__":

    
    routes_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/0',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/1',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/2',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/3',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/4',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/5',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/6',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/7',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/8',
                    'Rislab_Event_influence_volume/dataset/MNIST/Train/9']                  


    routes_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/0',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/1',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/2',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/3',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/4',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/5',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/6',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/7',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/8',
                    'Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/9'                   
                    ]

    main(path_images = routes_images, path_events = routes_events, class_name = "All")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/0'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/0'], class_name = "0")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/1'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/1'], class_name = "1")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/2'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/2'], class_name = "2")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/3'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/3'], class_name = "3")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/4'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/4'], class_name = "4")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/5'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/5'], class_name = "5")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/6'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/6'], class_name = "6")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/7'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/7'], class_name = "7")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/8'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/8'], class_name = "8")
    main(path_images = ['Rislab_Event_influence_volume/dataset/MNIST/Train/9'], path_events = ['Rislab_Event_influence_volume/dataset/N-MNIST/33ms/Train/9'], class_name = "9")

