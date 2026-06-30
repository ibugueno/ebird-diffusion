#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PANELS = (
    ("event_", "Input (event)"),
    ("generated_", "Generated"),
    ("target_", "Target"),
)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _index_images(input_dir: Path) -> dict[str, dict[str, Path]]:
    indexed: dict[str, dict[str, Path]] = {}
    for path in sorted(input_dir.glob("*.png")):
        for prefix, _ in PANELS:
            if path.name.startswith(prefix):
                suffix = path.name[len(prefix) :]
                indexed.setdefault(suffix, {})[prefix] = path
                break
    return indexed


def _compose(
    paths: list[Path],
    labels: list[str],
    *,
    include_labels: bool,
    label_height: int,
    gap: int,
) -> Image.Image:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    sizes = {image.size for image in images}
    if len(sizes) != 1:
        raise ValueError(
            "All images in a triplet must have the same dimensions: "
            + ", ".join(f"{path.name}={image.size}" for path, image in zip(paths, images))
        )

    width, height = images[0].size
    header = label_height if include_labels else 0
    canvas = Image.new(
        "RGB",
        (width * len(images) + gap * (len(images) - 1), height + header),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(max(12, label_height - 12))

    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * (width + gap)
        canvas.paste(image, (x, header))
        if include_labels:
            box = draw.textbbox((0, 0), label, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            draw.text(
                (x + (width - text_width) / 2, (header - text_height) / 2 - box[1]),
                label,
                fill="white",
                font=font,
            )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose Input | Generated | Target RGBE comparison images"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Maximum number of triplets")
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--label-height", type=int, default=44)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip incomplete triplets instead of failing",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    if args.gap < 0 or args.label_height < 1:
        raise ValueError("--gap must be non-negative and --label-height must be positive")

    indexed = _index_images(input_dir)
    complete: list[tuple[str, dict[str, Path]]] = []
    incomplete: list[str] = []
    expected = {prefix for prefix, _ in PANELS}
    for suffix, entries in sorted(indexed.items()):
        missing = expected - set(entries)
        if missing:
            incomplete.append(
                f"{suffix}: missing {', '.join(sorted(missing))}"
            )
        else:
            complete.append((suffix, entries))

    if incomplete and not args.skip_incomplete:
        preview = "\n".join(incomplete[:10])
        raise ValueError(
            f"Found {len(incomplete)} incomplete triplets. "
            f"Use --skip-incomplete to ignore them.\n{preview}"
        )
    if not complete:
        raise ValueError(f"No complete RGBE triplets found in {input_dir}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        complete = complete[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = [prefix for prefix, _ in PANELS]
    labels = [label for _, label in PANELS]
    for suffix, entries in complete:
        comparison = _compose(
            [entries[prefix] for prefix in prefixes],
            labels,
            include_labels=not args.no_labels,
            label_height=args.label_height,
            gap=args.gap,
        )
        comparison.save(output_dir / f"comparison_{suffix}")

    print(f"Created {len(complete)} comparison images in {output_dir}")
    if incomplete:
        print(f"Skipped {len(incomplete)} incomplete triplets")


if __name__ == "__main__":
    main()
