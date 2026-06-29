from __future__ import annotations

import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ebird.data import RGBEGazeDataset
from ebird.diffusion import LinearNoiseScheduler
from ebird.models.conditional import conditional_from_config
from ebird.models.unet import unet_from_config

from .distributed import DistributedContext, finish, initialize, mean_across_processes


Stage = Literal["image", "conditional"]


def _seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_logging(output_dir: Path, context: DistributedContext) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "training.log", mode="a"))
    logging.basicConfig(
        level=logging.INFO if context.is_main else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _dataset(config: dict, split: str, stage: Stage) -> RGBEGazeDataset:
    data = config["data"]
    return RGBEGazeDataset(
        Path(data["manifest_dir"]) / f"{split}.csv",
        data["dataset_root"],
        resolution=config["model"]["image_size"],
        include_condition=stage == "conditional",
    )


def _loader(
    dataset: RGBEGazeDataset,
    config: dict,
    context: DistributedContext,
    *,
    train_mode: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    training = config["training"]
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=train_mode,
            drop_last=train_mode,
        )
        if context.enabled
        else None
    )
    workers = int(training.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size_per_gpu", 1)),
        shuffle=train_mode and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=train_mode,
    )
    return loader, sampler


def _load_base(config: dict, device: torch.device) -> nn.Module:
    base = unet_from_config(config["model"])
    checkpoint_path = Path(config["training"]["base_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint base: {checkpoint_path}. "
            "Ejecuta primero --stage image."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base.load_state_dict(checkpoint["model_state_dict"])
    return base.to(device)


def _create_model(stage: Stage, config: dict, device: torch.device) -> nn.Module:
    if stage == "image":
        return unet_from_config(config["model"]).to(device)
    base = _load_base(config, device)
    return conditional_from_config(base, config["model"]).to(device)


def _raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _checkpoint_state(
    stage: Stage,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_loss: float,
    config: dict,
) -> dict:
    raw = _raw_model(model)
    state = {
        "stage": stage,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_loss": best_loss,
        "config": config,
    }
    if stage == "image":
        state["model_state_dict"] = raw.state_dict()
    else:
        state["control_state_dict"] = raw.control.state_dict()
    return state


def _restore(
    path: Path,
    stage: Stage,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[int, float]:
    if not path.is_file():
        return 0, float("inf")
    checkpoint = torch.load(path, map_location="cpu")
    raw = _raw_model(model)
    if stage == "image":
        raw.load_state_dict(checkpoint["model_state_dict"])
    else:
        raw.control.load_state_dict(checkpoint["control_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_loss", float("inf")))


def _forward(model: nn.Module, batch: dict, noisy: torch.Tensor, timesteps: torch.Tensor, stage: Stage):
    if stage == "image":
        return model(noisy, timesteps)
    return model(noisy, batch["condition"], timesteps)


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: LinearNoiseScheduler,
    stage: Stage,
    config: dict,
    context: DistributedContext,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    max_steps = config["training"].get("max_steps_per_epoch")
    mixed_precision = bool(config["training"].get("mixed_precision", True)) and context.device.type == "cuda"
    gradient_clip = config["training"].get("gradient_clip")
    losses: list[float] = []
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, disable=not context.is_main, leave=False)
    effective_length = len(loader) if max_steps is None else min(len(loader), int(max_steps))
    for step, batch in enumerate(progress):
        if max_steps is not None and step >= int(max_steps):
            break
        target = batch["target"].to(context.device, non_blocking=True)
        if stage == "conditional":
            batch["condition"] = batch["condition"].to(context.device, non_blocking=True)
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0, scheduler.num_timesteps, (target.shape[0],), device=context.device
        )
        noisy = scheduler.add_noise(target, noise, timesteps)
        should_step = (step + 1) % accumulation == 0 or step + 1 == effective_length
        sync_context = (
            model.no_sync()
            if train_mode
            and isinstance(model, DistributedDataParallel)
            and not should_step
            else nullcontext()
        )
        grad_context = torch.enable_grad() if train_mode else torch.no_grad()
        with grad_context, sync_context:
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                prediction = _forward(model, batch, noisy, timesteps, stage)
                loss = torch.nn.functional.mse_loss(prediction, noise)
                scaled_loss = loss / accumulation
            if train_mode:
                scaler.scale(scaled_loss).backward()

        if train_mode and should_step:
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().item()))
        if context.is_main:
            progress.set_postfix(loss=f"{losses[-1]:.4f}")

    local_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_across_processes(local_loss, context)


def train(stage: Stage, config: dict, *, resume: bool = True) -> None:
    if stage not in ("image", "conditional"):
        raise ValueError(f"Etapa inválida: {stage}")
    context = initialize()
    training = config["training"]
    output_dir = Path(training["output_dir"]) / stage
    _setup_logging(output_dir, context)
    _seed(int(training.get("seed", 44)), context.rank)

    try:
        train_dataset = _dataset(config, "train", stage)
        train_loader, train_sampler = _loader(
            train_dataset, config, context, train_mode=True
        )
        val_loader = None
        val_path = Path(config["data"]["manifest_dir"]) / "val.csv"
        if val_path.is_file() and val_path.stat().st_size > 0:
            try:
                val_dataset = _dataset(config, "val", stage)
                val_loader, _ = _loader(val_dataset, config, context, train_mode=False)
            except ValueError:
                val_loader = None

        model = _create_model(stage, config, context.device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(training.get("mixed_precision", True))
            and context.device.type == "cuda"
        )
        if context.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )

        checkpoint_dir = output_dir / "checkpoints"
        last_checkpoint = checkpoint_dir / "last.pt"
        start_epoch, best_loss = (
            _restore(last_checkpoint, stage, model, optimizer, scaler)
            if resume
            else (0, float("inf"))
        )
        epochs = int(
            training["image_epochs"] if stage == "image" else training["conditional_epochs"]
        )
        scheduler = LinearNoiseScheduler(**config["diffusion"])
        writer = SummaryWriter(output_dir / "tensorboard") if context.is_main else None
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_config.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            logging.info(
                "Etapa=%s | dispositivo=%s | procesos=%d | muestras=%d | parámetros entrenables=%d",
                stage,
                context.device,
                context.world_size,
                len(train_dataset),
                sum(parameter.numel() for parameter in trainable),
            )

        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = _run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                stage=stage,
                config=config,
                context=context,
            )
            val_loss = (
                _run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    scaler=scaler,
                    scheduler=scheduler,
                    stage=stage,
                    config=config,
                    context=context,
                )
                if val_loader is not None
                else train_loss
            )
            if context.is_main:
                logging.info(
                    "Epoch %d/%d | train=%.6f | val=%.6f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )
                if writer:
                    writer.add_scalar("loss/train", train_loss, epoch)
                    writer.add_scalar("loss/val", val_loss, epoch)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_loss = min(best_loss, val_loss)
                state = _checkpoint_state(
                    stage, model, optimizer, scaler, epoch, best_loss, config
                )
                torch.save(state, last_checkpoint)
                if val_loss <= best_loss:
                    torch.save(state, checkpoint_dir / "best.pt")
            if context.enabled:
                dist.barrier()
        if writer:
            writer.close()
    finally:
        finish(context)
