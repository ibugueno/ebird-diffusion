#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MODEL_ROWS = (
    (
        "ddpm-512-user1",
        "DDPM 512 - user 1 (baseline)",
        Path("ddpm-512-user1/user_1/exp6_comparisons"),
    ),
    (
        "ddpm-512-user1-v2-40",
        "DDPM 512 - user 1 (V2, 40 epochs)",
        Path("ddpm-512-user1-v2-40/user_1/exp6_comparisons"),
    ),
    (
        "pix2pix-512-user1",
        "Pix2Pix 512 - user 1",
        Path("pix2pix-512-user1/user_1/exp6_comparisons"),
    ),
)


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Comparison directory does not exist: {directory}")
    images = {path.name: path for path in sorted(directory.glob("comparison_*.png"))}
    if not images:
        raise ValueError(f"No comparison PNG files found in {directory}")
    return images


def add_row_title(
    image: Image.Image,
    title: str,
    *,
    title_height: int,
) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + title_height), "black")
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(12, title_height - 14))
    box = draw.textbbox((0, 0), title, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    draw.text(
        (
            (image.width - text_width) / 2,
            (title_height - text_height) / 2 - box[1],
        ),
        title,
        fill="white",
        font=font,
    )
    return canvas


def stack_rows(rows: list[Image.Image], *, gap: int) -> Image.Image:
    widths = {row.width for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Row widths do not match: {sorted(widths)}")
    width = rows[0].width
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "black")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack DDPM baseline, DDPM V2-40, and Pix2Pix comparisons"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.home() / "tmp4",
        help="Directory containing the three model folders",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--title-height", type=int, default=44)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.title_height < 1 or args.gap < 0:
        raise ValueError("--title-height must be positive and --gap non-negative")

    root_dir = args.root_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root_dir / "stacked-model-comparisons"
    )
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another --output-dir or add --overwrite."
        )

    indexed = [
        (key, title, image_map(root_dir / relative_dir))
        for key, title, relative_dir in MODEL_ROWS
    ]
    all_names = set().union(*(set(images) for _, _, images in indexed))
    complete_names = sorted(
        name
        for name in all_names
        if all(name in images for _, _, images in indexed)
    )
    incomplete_names = sorted(all_names - set(complete_names))
    if incomplete_names and not args.skip_incomplete:
        preview = ", ".join(incomplete_names[:5])
        raise ValueError(
            f"Found {len(incomplete_names)} filenames missing from at least one model. "
            f"Examples: {preview}. Add --skip-incomplete to ignore them."
        )
    if not complete_names:
        raise ValueError("No filenames are shared by all three model folders")
    if args.limit is not None:
        complete_names = complete_names[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in complete_names:
        titled_rows: list[Image.Image] = []
        expected_size: tuple[int, int] | None = None
        for _, title, images in indexed:
            with Image.open(images[name]) as image:
                current = image.convert("RGB")
            if expected_size is None:
                expected_size = current.size
            elif current.size != expected_size:
                raise ValueError(
                    f"Comparison dimensions differ for {name}: "
                    f"expected {expected_size}, found {current.size}"
                )
            titled_rows.append(
                add_row_title(current, title, title_height=args.title_height)
            )
        stack_rows(titled_rows, gap=args.gap).save(
            output_dir / f"stack_{name}"
        )

    print(f"Created {len(complete_names)} vertical stacks in {output_dir}")
    if incomplete_names:
        print(f"Skipped {len(incomplete_names)} incomplete comparisons")


if __name__ == "__main__":
    main()

