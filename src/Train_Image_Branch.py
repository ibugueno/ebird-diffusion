#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tensorboard
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Módulos del usuario
from UnetClass2 import Unet
from Scheduler import LinearNoiseScheduler
from Datasets import SingleImageDataset


def seed_everything(seed: int = 44):
    """Fija la semilla para reproducibilidad."""
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Path, level=logging.INFO):
    """Configura logging a archivo y consola."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ("train_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging to %s", log_file)


def save_checkpoint(state: dict, ckpt_dir: Path, is_best: bool = False, ckpt_name: str = 'ddpm_ckpt.pth'):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / ckpt_name
    torch.save(state, last_path)
    if is_best:
        best_path = ckpt_dir / ckpt_name
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
    else:
        logging.warning("No se encontró checkpoint en %s", ckpt_path)
        return 0, float("inf")

# ---------------------------------- bucles de entrenamiento ------------------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, grad_clip=None):
    model.train()
    epoch_losses = []
    for im in tqdm(loader, desc="Train", leave=False):
        im = im.float().to(device, non_blocking=True)
        noise = torch.randn_like(im)
        timesteps = torch.randint(0, scheduler.num_timesteps, (im.shape[0],), device=device)

        noisy_im = scheduler.add_noise(im, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            noise_pred = model(noisy_im, timesteps)
            loss = criterion(noise_pred, noise)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(loss.item())
    return float(np.mean(epoch_losses))


def validate(model, loader, criterion, scheduler, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for im in tqdm(loader, desc="Val", leave=False):
            im = im.float().to(device, non_blocking=True)
            noise = torch.randn_like(im)
            timesteps = torch.randint(0, scheduler.num_timesteps, (im.shape[0],), device=device)
            noisy_im = scheduler.add_noise(im, noise, timesteps)
            noise_pred = model(noisy_im, timesteps)
            loss = criterion(noise_pred, noise)
            val_losses.append(loss.item())
    return float(np.mean(val_losses))

# ---------------------------------- función principal ------------------------------------------ #

def main(cfg_path: str, resume: bool = True):
    # 1) Config ------------------------------------------------------------------
    with open(cfg_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error al leer YAML: {exc}")

    diffusion_cfg = cfg["diffusion_params"]
    data_cfg = cfg["dataset_params"]
    model_cfg = cfg["model_params"]
    train_cfg = cfg["train_params"]
    save_name = train_cfg["ckpt_name"]

    seed_everything(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # velocidad si las dimensiones son fijas

    # 2) Logging ------------------------------------------------------------------
    run_dir = Path(train_cfg["task_name"], "DDPM")
    setup_logger(run_dir)
    writer = SummaryWriter(log_dir=run_dir / "tb")

    # 3) Dataset ------------------------------------------------------------------
    train_ds = SingleImageDataset(im_paths=data_cfg["paths"])
    train_loader = DataLoader(
        train_ds, # 1. Dataset de entrada
        batch_size=train_cfg["batch_size"],  # 2. Tamaño del batch
        shuffle=True, # 3. Mezclar los datos en cada época
        num_workers=4,  # 4. Número de procesos para cargar datos
        pin_memory=(device.type == "cuda"), # 5. Optimiza la transferencia a GPU
        #Hace que la transferencia a GPU sea más rápida, ya que los datos se almacenan en una zona especial de la RAM (pinneada).
        #Sin esto, .to(device) puede ser más lento.

        persistent_workers=True, # 6. Mantiene procesos abiertos entre épocas
    )

    # Si tienes validación separa aquí dataset.
    val_loader = None  # sustituye en caso de tener val

    # 4) Modelo, scheduler, optim --------------------------------------------------
    model = Unet(model_cfg).to(device)
    scheduler_noise = LinearNoiseScheduler(
        num_timesteps=diffusion_cfg["num_timesteps"],
        beta_start=diffusion_cfg["beta_start"],
        beta_end=diffusion_cfg["beta_end"],
    )

    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], betas=(0.9, 0.999), weight_decay=train_cfg.get("weight_decay", 0.0))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.MSELoss()

    # 5) Checkpointing ------------------------------------------------------------
    ckpt_dir = run_dir / "checkpoints"
    start_epoch, best_loss = (0, float("inf"))
    if resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, ckpt_dir / save_name)

    # 6) Bucle de entrenamiento ----------------------------------------------------
    num_epochs = train_cfg["num_epochs"]
    grad_clip = train_cfg.get("grad_clip", None)

    try:
        for epoch in range(start_epoch, num_epochs):
            logging.info("\n Epoch %d/%d", epoch + 1, num_epochs)
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler_noise, device, grad_clip)
            writer.add_scalar("Loss/train", train_loss, epoch)
            logging.info("Train loss: %.6f", train_loss)

            # Validación opcional
            if val_loader is not None:
                val_loss = validate(model, val_loader, criterion, scheduler_noise, device)
                writer.add_scalar("Loss/val", val_loss, epoch)
                logging.info("Val loss  : %.6f", val_loss)
            else:
                val_loss = train_loss  # si no hay val usa train para best

            # Checkpointing
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
                ckpt_dir,
                is_best=is_best,
                ckpt_name=save_name
            )

    except KeyboardInterrupt:
        logging.warning("Entrenamiento interrumpido por el usuario. Guardando estado...")
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            ckpt_dir,
            ckpt_name=save_name
        )
    finally:
        writer.close()
        logging.info("Fin del entrenamiento")

# ---------------------------------- CLI -------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento DDPM para rama de imágenes")
    parser.add_argument("--config", dest="config_path", default="src/default.yaml", type=str, help="Ruta al archivo YAML de configuración")
    parser.add_argument("--no-resume", action="store_true", help="No reanudar desde el último checkpoint")
    args = parser.parse_args()

    main(args.config_path, resume=True)
    #main(args.config_path, resume=not args.no_resume)
