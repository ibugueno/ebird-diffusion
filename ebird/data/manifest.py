from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = (
    "target_path",
    "event_path",
    "user",
    "experiment",
    "filename",
    "split",
)
USER_PATTERN = re.compile(r"^user_(\d+)$", re.IGNORECASE)
EXPERIMENT_PATTERN = re.compile(r"^exp(?:eriment)?_?(\d+)$", re.IGNORECASE)


def _image_map(root: Path) -> dict[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de imágenes: {root}")
    images: dict[Path, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(root)
            if relative in images:
                raise ValueError(f"Ruta relativa duplicada: {relative}")
            images[relative] = path
    return images


def _parse_identity(relative_path: Path) -> tuple[str, str]:
    user = "unknown"
    experiment = "unknown"
    for part in relative_path.parts[:-1]:
        user_match = USER_PATTERN.match(part)
        experiment_match = EXPERIMENT_PATTERN.match(part)
        if user_match:
            user = f"user_{int(user_match.group(1))}"
        if experiment_match:
            experiment = f"exp{int(experiment_match.group(1))}"
    if user == "unknown":
        raise ValueError(
            f"No se pudo inferir el usuario desde la ruta: {relative_path}"
        )
    return user, experiment


def _numeric_user_key(user: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", user)
    return (int(match.group(1)) if match else 10**9, user)


def _normalize_user(user: str) -> str:
    value = str(user).strip().lower()
    match = re.fullmatch(r"(?:user_?)?(\d+)", value)
    return f"user_{int(match.group(1))}" if match else value


def split_users(
    users: Iterable[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique_users = sorted(set(users), key=_numeric_user_key)
    if not unique_users:
        raise ValueError("No se encontraron usuarios")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Los ratios de validación y test deben sumar menos de 1")

    random.Random(seed).shuffle(unique_users)
    count = len(unique_users)
    if count == 1:
        return {unique_users[0]: "train"}

    test_count = max(1, round(count * test_ratio))
    val_count = max(1, round(count * val_ratio)) if count >= 3 and val_ratio > 0 else 0
    while count - test_count - val_count < 1:
        if val_count:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    assignment: dict[str, str] = {}
    for user in unique_users[:test_count]:
        assignment[user] = "test"
    for user in unique_users[test_count : test_count + val_count]:
        assignment[user] = "val"
    for user in unique_users[test_count + val_count :]:
        assignment[user] = "train"
    return assignment


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    target_dir: str = "gray_frames",
    event_dir: str = "event_accumulate_frames",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 44,
    include_users: Iterable[str] | None = None,
    strict_pairs: bool = False,
) -> dict[str, int]:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    targets = _image_map(dataset_root / target_dir)
    events = _image_map(dataset_root / event_dir)

    selected_users = (
        {_normalize_user(user) for user in include_users} if include_users else None
    )
    if selected_users:
        available_users = {
            _parse_identity(relative)[0] for relative in set(targets) | set(events)
        }
        missing_users = sorted(selected_users - available_users, key=_numeric_user_key)
        if missing_users:
            raise ValueError(
                "No se encontraron los usuarios solicitados: " + ", ".join(missing_users)
            )
        targets = {
            relative: path
            for relative, path in targets.items()
            if _parse_identity(relative)[0] in selected_users
        }
        events = {
            relative: path
            for relative, path in events.items()
            if _parse_identity(relative)[0] in selected_users
        }

    missing_events = sorted(set(targets) - set(events))
    missing_targets = sorted(set(events) - set(targets))
    if strict_pairs and (missing_events or missing_targets):
        examples = [str(path) for path in (missing_events + missing_targets)[:10]]
        raise ValueError(
            "El dataset no está completamente emparejado. "
            f"Sin evento: {len(missing_events)}; sin frame: {len(missing_targets)}; "
            f"ejemplos: {examples}"
        )

    paired_paths = sorted(set(targets) & set(events))
    if not paired_paths:
        raise ValueError("No se encontraron pares frame/evento válidos")

    rows: list[dict[str, str]] = []
    for relative in paired_paths:
        user, experiment = _parse_identity(relative)
        rows.append(
            {
                "target_path": str(targets[relative].relative_to(dataset_root)),
                "event_path": str(events[relative].relative_to(dataset_root)),
                "user": user,
                "experiment": experiment,
                "filename": relative.name,
                "split": "",
            }
        )

    assignments = split_users(
        (row["user"] for row in rows),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for row in rows:
        row["split"] = assignments[row["user"]]

    _write_manifest(output_dir / "all.csv", rows)
    counts = Counter(row["split"] for row in rows)
    for split in ("train", "val", "test"):
        _write_manifest(
            output_dir / f"{split}.csv",
            [row for row in rows if row["split"] == split],
        )
    return {
        "paired": len(rows),
        "skipped_without_event": len(missing_events),
        "skipped_without_target": len(missing_targets),
        **{split: counts.get(split, 0) for split in ("train", "val", "test")},
    }


def validate_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    verify_images: bool = True,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors: list[str] = []
    users: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        users.add(row.get("user", "unknown"))
        for field in ("target_path", "event_path"):
            path = dataset_root / row[field]
            if not path.is_file():
                errors.append(f"línea {row_index}: no existe {path}")
                continue
            if verify_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:  # Pillow expone varias excepciones.
                    errors.append(f"línea {row_index}: imagen inválida {path}: {exc}")
        if len(errors) >= 50:
            break
    return {
        "rows": len(rows),
        "users": len(users),
        "valid": not errors,
        "errors": errors,
    }
