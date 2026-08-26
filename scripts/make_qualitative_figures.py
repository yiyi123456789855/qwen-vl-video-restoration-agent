import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在：{path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def load_metrics(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sequence_rows(metrics, mode):
    return [
        row for row in metrics["sequences"] if row["mode"] == mode
    ]


def representative_noisy_rows(metrics, count=3):
    rows = [
        row for row in sequence_rows(metrics, "noisy")
        if row["sequence"].endswith("srcavi_pair")
    ]
    if not rows:
        rows = sequence_rows(metrics, "noisy")

    # PSNR低、中、高各取一个，避免重复展示同一噪声强度。
    rows.sort(key=lambda row: row["valid_original"]["psnr"])
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[index] for index in indices]


def condition_key(sequence):
    match = re.match(r"(scene\d+__ISO\d+)__", sequence)
    return match.group(1) if match else sequence


def unique_false_triggers(metrics):
    wrong = [
        row for row in sequence_rows(metrics, "clean")
        if not row["tool_correct"]
    ]
    unique = {}
    for row in wrong:
        unique.setdefault(condition_key(row["sequence"]), row)
    return list(unique.values())


def amplified_difference(left: Image.Image, right: Image.Image, scale):
    left_array = np.asarray(left, dtype=np.int16)
    right_array = np.asarray(right, dtype=np.int16)
    difference = np.abs(left_array - right_array) * scale
    difference = np.clip(difference, 0, 255).astype(np.uint8)
    return Image.fromarray(difference, mode="RGB")


def fit_panel(image: Image.Image, panel_width, panel_height):
    result = image.copy()
    result.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_width, panel_height), "white")
    left = (panel_width - result.width) // 2
    top = (panel_height - result.height) // 2
    canvas.paste(result, (left, top))
    return canvas


def make_grid(rows, output_path: Path, panel_width=420, panel_height=320):
    label_height = 52
    margin = 14
    columns = max(len(row) for row in rows)
    width = margin + columns * (panel_width + margin)
    height = margin + len(rows) * (panel_height + label_height + margin)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    for row_index, row in enumerate(rows):
        top = margin + row_index * (panel_height + label_height + margin)
        for column_index, (label, image) in enumerate(row):
            left = margin + column_index * (panel_width + margin)
            panel = fit_panel(image, panel_width, panel_height)
            canvas.paste(panel, (left, top))
            draw.multiline_text(
                (left + 4, top + panel_height + 5),
                label,
                fill="black",
                font=font,
                spacing=3,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def choose_existing_frame(base_paths, requested_name):
    requested = [base / requested_name for base in base_paths]
    if all(path.is_file() for path in requested):
        return requested_name

    name_sets = []
    for base in base_paths:
        name_sets.append(
            {path.name for path in base.iterdir() if path.is_file()}
        )
    common = sorted(set.intersection(*name_sets))
    if not common:
        raise RuntimeError(f"没有可配对图像：{base_paths}")
    return common[len(common) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--qlora_pipeline_root", required=True)
    parser.add_argument("--zero_pipeline_root", required=True)
    parser.add_argument("--qlora_metrics", required=True)
    parser.add_argument("--zero_metrics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame", default="000007.png")
    parser.add_argument("--difference_scale", type=float, default=20.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    qlora_root = Path(args.qlora_pipeline_root).expanduser().resolve()
    zero_root = Path(args.zero_pipeline_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    qlora_metrics = load_metrics(
        Path(args.qlora_metrics).expanduser().resolve()
    )
    zero_metrics = load_metrics(
        Path(args.zero_metrics).expanduser().resolve()
    )

    noisy_panels = []
    selected_noisy = representative_noisy_rows(qlora_metrics)
    for row in selected_noisy:
        sequence = row["sequence"]
        input_dir = dataset_root / "test" / "noisy" / sequence
        restored_dir = qlora_root / "noisy" / sequence / "restored"
        target_dir = dataset_root / "test" / "clean" / sequence
        frame = choose_existing_frame(
            [input_dir, restored_dir, target_dir], args.frame
        )
        noisy = load_rgb(input_dir / frame)
        restored = load_rgb(restored_dir / frame)
        target = load_rgb(target_dir / frame)
        difference = amplified_difference(restored, target, 5.0)
        noisy_panels.append(
            [
                (f"Noisy input\n{sequence}", noisy),
                (
                    "QLoRA pipeline\n"
                    f"Seq.avg PSNR={row['valid_restored']['psnr']:.2f}, "
                    f"SSIM={row['valid_restored']['ssim']:.4f}",
                    restored,
                ),
                ("Ground truth", target),
                ("|Restored - GT| x5", difference),
            ]
        )

    noisy_path = output_dir / "noisy_restoration_cases.png"
    make_grid(noisy_panels, noisy_path)

    false_triggers = unique_false_triggers(zero_metrics)
    if not false_triggers:
        raise RuntimeError("Zero-shot结果中没有干净图误触发案例")
    false_triggers.sort(key=lambda row: row["valid_restored"]["psnr"])
    row = false_triggers[0]
    sequence = row["sequence"]
    clean_dir = dataset_root / "test" / "clean" / sequence
    zero_dir = zero_root / "clean" / sequence / "restored"
    qlora_dir = qlora_root / "clean" / sequence / "restored"
    frame = choose_existing_frame(
        [clean_dir, zero_dir, qlora_dir], args.frame
    )
    clean = load_rgb(clean_dir / frame)
    zero_output = load_rgb(zero_dir / frame)
    qlora_output = load_rgb(qlora_dir / frame)
    difference = amplified_difference(
        clean,
        zero_output,
        args.difference_scale,
    )
    clean_path = output_dir / "clean_false_trigger_case.png"
    make_grid(
        [
            [
                (f"Clean input\n{sequence}", clean),
                (
                    "Zero-shot: denoise\n"
                    f"Seq.avg PSNR={row['valid_restored']['psnr']:.2f}, "
                    f"SSIM={row['valid_restored']['ssim']:.4f}",
                    zero_output,
                ),
                (
                    f"|Input - Zero-shot| x{args.difference_scale:g}",
                    difference,
                ),
                ("QLoRA: passthrough\nPixel MAE=0", qlora_output),
            ]
        ],
        clean_path,
    )

    manifest = {
        "noisy_cases": [row["sequence"] for row in selected_noisy],
        "clean_false_trigger": sequence,
        "frame": frame,
        "difference_scale": args.difference_scale,
        "outputs": [str(noisy_path), str(clean_path)],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
