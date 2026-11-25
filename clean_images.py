from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (128, 128)
BACKGROUND_GRAY = 127


def expand_path(value: str | Path) -> Path:
    def environment_value(name: str, original: str) -> str:
        value = os.environ.get(name)
        if value is not None:
            return value
        if name.upper() in {"HOME", "USERPROFILE"}:
            return str(Path.home())
        return original

    raw_path = os.path.expandvars(str(value))
    raw_path = re.sub(
        r"\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda match: environment_value(match.group(1), match.group(0)),
        raw_path,
    )
    raw_path = re.sub(
        r"%([^%]+)%",
        lambda match: environment_value(match.group(1), match.group(0)),
        raw_path,
    )
    return Path(raw_path).expanduser()


def image_files(root: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (BACKGROUND_GRAY,) * 3 + (255,))
            image = Image.alpha_composite(background, rgba)
        return image.convert("L")


def prepare_image(image: Image.Image, mode: str, threshold: int) -> Image.Image:
    fitted = ImageOps.contain(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", IMAGE_SIZE, color=BACKGROUND_GRAY)
    offset = ((IMAGE_SIZE[0] - fitted.width) // 2, (IMAGE_SIZE[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)

    if mode == "binary":
        return canvas.point(lambda pixel: 255 if pixel >= threshold else 0)
    return canvas


def difference_hash(image: Image.Image) -> int:
    small = image.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | (left > right)
    return value


def hamming_distance(first: int, second: int) -> int:
    return bin(first ^ second).count("1")


def unique_output_path(output_dir: Path, relative_path: Path) -> Path:
    output_path = output_dir / relative_path.with_suffix(".png")
    if not output_path.exists():
        return output_path

    suffix = 2
    while True:
        candidate = output_path.with_name(f"{output_path.stem}__{suffix}.png")
        if not candidate.exists():
            return candidate
        suffix += 1


def process_images(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    threshold: int,
    duplicate_distance: int,
) -> tuple[int, int, int, int]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("The input and output directories cannot overlap.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"The output directory already contains files: {output_dir}. "
            "Choose another directory with --output-dir to avoid overwriting results."
        )

    files = list(image_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No supported images were found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_exact: dict[str, dict[tuple[str, ...], Path]] = {}
    seen_hashes: list[tuple[int, Path, str]] = []
    counts = {"processed": 0, "duplicate_skipped": 0, "similar": 0, "invalid": 0}
    report_path = output_dir / "cleaning_report.csv"

    with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
        fields = [
            "source_file",
            "output_file",
            "status",
            "exact_duplicate_of",
            "possible_similar_to",
            "hash_distance",
            "error",
        ]
        writer = csv.DictWriter(report_file, fieldnames=fields)
        writer.writeheader()

        for source_path in files:
            relative_path = source_path.relative_to(input_dir)
            scope = relative_path.parts[:-1]
            row = {
                "source_file": str(relative_path),
                "output_file": "",
                "status": "",
                "exact_duplicate_of": "",
                "possible_similar_to": "",
                "hash_distance": "",
                "error": "",
            }

            try:
                file_hash = sha256_file(source_path)
                prepared = prepare_image(load_grayscale(source_path), mode, threshold)
                visual_hash = difference_hash(prepared)
            except (OSError, ValueError) as error:
                row["status"] = "invalid_image"
                row["error"] = str(error)
                writer.writerow(row)
                counts["invalid"] += 1
                continue

            prior_exact = seen_exact.get(file_hash)
            similar_matches = [
                (distance, prior_path)
                for prior_hash, prior_path, prior_file_hash in seen_hashes
                if (distance := hamming_distance(visual_hash, prior_hash))
                <= duplicate_distance
                and prior_file_hash != file_hash
            ]
            if similar_matches:
                distance, prior_path = min(similar_matches, key=lambda match: match[0])
                row["possible_similar_to"] = str(prior_path.relative_to(input_dir))
                row["hash_distance"] = str(distance)
                counts["similar"] += 1

            exact_in_scope = seen_exact.get(file_hash, {})
            exact_elsewhere = next(
                (prior_path for prior_scope, prior_path in exact_in_scope.items() if prior_scope != scope),
                None,
            )
            if scope in exact_in_scope:
                row["status"] = "exact_duplicate_skipped"
                row["exact_duplicate_of"] = str(exact_in_scope[scope].relative_to(input_dir))
                counts["duplicate_skipped"] += 1
            else:
                output_path = unique_output_path(output_dir, relative_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prepared.save(output_path, format="PNG", optimize=True)
                row["output_file"] = str(output_path.relative_to(output_dir))
                if exact_elsewhere is not None:
                    row["status"] = "exact_duplicate_in_other_folder_kept_for_review"
                    row["exact_duplicate_of"] = str(exact_elsewhere.relative_to(input_dir))
                elif similar_matches:
                    row["status"] = "possible_similar_kept_for_review"
                else:
                    row["status"] = "processed"
                counts["processed"] += 1

            seen_exact.setdefault(file_hash, {}).setdefault(scope, source_path)
            seen_hashes.append((visual_hash, source_path, file_hash))
            writer.writerow(row)

    print(f"Images found: {len(files)}")
    print(f"Images processed for the copy: {counts['processed']}")
    print(f"Exact duplicates skipped in the same folder/class: {counts['duplicate_skipped']}")
    print(f"Possible similar images to review: {counts['similar']}")
    print(f"Invalid images: {counts['invalid']}")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")
    return counts["processed"], counts["duplicate_skipped"], counts["similar"], counts["invalid"]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Standardizes images as 128x128 PNGs, detects duplicates, and "
            "converts them to grayscale."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=expand_path,
        default=project_dir / "data",
        help=(
            "Image directory. Can be absolute, contain spaces, ~, $HOME, "
            "or %%USERPROFILE%% (default: data next to the script)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=expand_path,
        default=project_dir / "processed_data",
        help="New directory for results (default: processed_data next to the script).",
    )
    parser.add_argument(
        "--mode",
        choices=("grayscale", "binary"),
        default="grayscale",
        help="grayscale preserves gray tones; binary produces black or white pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Threshold from 0 to 255, used only with --mode binary.",
    )
    parser.add_argument(
        "--duplicate-distance",
        type=int,
        default=4,
        help="Maximum visual hash distance for flagging similar images (0 to 64).",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255.")
    if not 0 <= args.duplicate_distance <= 64:
        parser.error("--duplicate-distance must be between 0 and 64.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    process_images(
        arguments.input_dir,
        arguments.output_dir,
        arguments.mode,
        arguments.threshold,
        arguments.duplicate_distance,
    )
