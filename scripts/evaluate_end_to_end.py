import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim


IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


def natural_key(path: Path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def list_images(folder: Path):
    return sorted(
        [
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_key,
    )


def load_gray(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L"),
            dtype=np.float32,
        )
    return torch.from_numpy(array.copy()).unsqueeze(0).div_(255.0)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction, target).item()
    if mse <= 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def ssim_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        ssim(
            prediction.unsqueeze(0),
            target.unsqueeze(0),
            data_range=1.0,
            size_average=True,
        ).item()
    )


def mean(values):
    return sum(values) / len(values) if values else 0.0


def evaluate_frames(predictions, targets):
    psnrs = [
        psnr(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]
    ssims = [
        ssim_value(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]
    temporal = []
    for index in range(1, len(predictions)):
        pred_difference = predictions[index] - predictions[index - 1]
        target_difference = targets[index] - targets[index - 1]
        temporal.append(
            float(
                torch.mean(
                    torch.abs(pred_difference - target_difference)
                ).item()
            )
        )

    return {
        "psnr": mean(psnrs),
        "ssim": mean(ssims),
        "temporal_difference_error": mean(temporal),
        "frames": len(predictions),
        "temporal_pairs": len(temporal),
    }


def rounded_metrics(metrics):
    return {
        key: round(value, 8) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def evaluate_sequence(
    mode: str,
    sequence: str,
    input_dir: Path,
    target_dir: Path,
    pipeline_dir: Path,
    radius: int,
):
    report_path = pipeline_dir / "run_report.json"
    restored_dir = pipeline_dir / "restored"
    if not report_path.is_file():
        raise FileNotFoundError(f"缺少报告：{report_path}")
    if not restored_dir.is_dir():
        raise FileNotFoundError(f"缺少复原目录：{restored_dir}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    input_paths = list_images(input_dir)
    target_by_name = {path.name: path for path in list_images(target_dir)}
    restored_by_name = {
        path.name: path for path in list_images(restored_dir)
    }

    missing_targets = [
        path.name for path in input_paths if path.name not in target_by_name
    ]
    missing_outputs = [
        path.name for path in input_paths if path.name not in restored_by_name
    ]
    if missing_targets or missing_outputs:
        raise RuntimeError(
            f"{mode}/{sequence} 配对不完整，"
            f"missing_targets={missing_targets}, "
            f"missing_outputs={missing_outputs}"
        )

    inputs = [load_gray(path) for path in input_paths]
    targets = [load_gray(target_by_name[path.name]) for path in input_paths]
    restored = [
        load_gray(restored_by_name[path.name]) for path in input_paths
    ]

    all_original = evaluate_frames(inputs, targets)
    all_restored = evaluate_frames(restored, targets)

    if len(input_paths) > radius * 2:
        valid_slice = slice(radius, len(input_paths) - radius)
        valid_original = evaluate_frames(
            inputs[valid_slice],
            targets[valid_slice],
        )
        valid_restored = evaluate_frames(
            restored[valid_slice],
            targets[valid_slice],
        )
    else:
        valid_original = all_original
        valid_restored = all_restored

    exact_copies = sum(
        input_path.read_bytes()
        == restored_by_name[input_path.name].read_bytes()
        for input_path in input_paths
    )
    pixel_mae = mean(
        [
            float(torch.mean(torch.abs(source - result)).item())
            for source, result in zip(inputs, restored)
        ]
    )

    expected_tool = "denoise" if mode == "noisy" else "none"
    predicted_tool = report.get("selected_tool", "unknown")

    return {
        "mode": mode,
        "sequence": sequence,
        "expected_tool": expected_tool,
        "predicted_tool": predicted_tool,
        "tool_correct": predicted_tool == expected_tool,
        "action": report.get("action"),
        "frames": len(input_paths),
        "exact_copy_rate": exact_copies / len(input_paths),
        "input_output_pixel_mae": pixel_mae,
        "all_original": rounded_metrics(all_original),
        "all_restored": rounded_metrics(all_restored),
        "valid_original": rounded_metrics(valid_original),
        "valid_restored": rounded_metrics(valid_restored),
    }


def aggregate(rows, metric_group):
    return {
        "psnr": mean([row[metric_group]["psnr"] for row in rows]),
        "ssim": mean([row[metric_group]["ssim"] for row in rows]),
        "temporal_difference_error": mean(
            [
                row[metric_group]["temporal_difference_error"]
                for row in rows
            ]
        ),
    }


def summarize(rows):
    summary = {}
    for mode in ("noisy", "clean"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        if not mode_rows:
            continue

        all_original = aggregate(mode_rows, "all_original")
        all_restored = aggregate(mode_rows, "all_restored")
        valid_original = aggregate(mode_rows, "valid_original")
        valid_restored = aggregate(mode_rows, "valid_restored")

        summary[mode] = {
            "sequences": len(mode_rows),
            "frames": sum(row["frames"] for row in mode_rows),
            "tool_accuracy": mean(
                [float(row["tool_correct"]) for row in mode_rows]
            ),
            "selected_tool_counts": dict(
                Counter(row["predicted_tool"] for row in mode_rows)
            ),
            "exact_copy_rate": mean(
                [row["exact_copy_rate"] for row in mode_rows]
            ),
            "input_output_pixel_mae": mean(
                [row["input_output_pixel_mae"] for row in mode_rows]
            ),
            "all_frames": {
                "original": rounded_metrics(all_original),
                "pipeline": rounded_metrics(all_restored),
                "psnr_gain": round(
                    all_restored["psnr"] - all_original["psnr"], 8
                ),
                "ssim_gain": round(
                    all_restored["ssim"] - all_original["ssim"], 8
                ),
                "temporal_error_reduction": round(
                    all_original["temporal_difference_error"]
                    - all_restored["temporal_difference_error"],
                    8,
                ),
            },
            "valid_center_frames": {
                "original": rounded_metrics(valid_original),
                "pipeline": rounded_metrics(valid_restored),
                "psnr_gain": round(
                    valid_restored["psnr"] - valid_original["psnr"],
                    8,
                ),
                "ssim_gain": round(
                    valid_restored["ssim"] - valid_original["ssim"],
                    8,
                ),
                "temporal_error_reduction": round(
                    valid_original["temporal_difference_error"]
                    - valid_restored["temporal_difference_error"],
                    8,
                ),
            },
        }

    summary["overall_tool_accuracy"] = mean(
        [float(row["tool_correct"]) for row in rows]
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--pipeline_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in ("noisy", "clean"):
        input_root = dataset_root / "test" / mode
        target_root = dataset_root / "test" / "clean"
        for input_dir in sorted(
            path for path in input_root.iterdir() if path.is_dir()
        ):
            sequence = input_dir.name
            print(f"Evaluating {mode}/{sequence}", flush=True)
            rows.append(
                evaluate_sequence(
                    mode=mode,
                    sequence=sequence,
                    input_dir=input_dir,
                    target_dir=target_root / sequence,
                    pipeline_dir=pipeline_root / mode / sequence,
                    radius=args.radius,
                )
            )

    summary = summarize(rows)
    result = {
        "dataset_root": str(dataset_root),
        "pipeline_root": str(pipeline_root),
        "radius": args.radius,
        "summary": summary,
        "sequences": rows,
    }
    metrics_path = output_dir / "end_to_end_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "sequence_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "mode", "sequence", "expected_tool", "predicted_tool",
            "tool_correct", "frames", "exact_copy_rate",
            "input_output_pixel_mae", "original_psnr_valid",
            "pipeline_psnr_valid", "original_ssim_valid",
            "pipeline_ssim_valid", "original_temporal_valid",
            "pipeline_temporal_valid",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row["mode"],
                    "sequence": row["sequence"],
                    "expected_tool": row["expected_tool"],
                    "predicted_tool": row["predicted_tool"],
                    "tool_correct": row["tool_correct"],
                    "frames": row["frames"],
                    "exact_copy_rate": row["exact_copy_rate"],
                    "input_output_pixel_mae": row[
                        "input_output_pixel_mae"
                    ],
                    "original_psnr_valid": row["valid_original"]["psnr"],
                    "pipeline_psnr_valid": row["valid_restored"]["psnr"],
                    "original_ssim_valid": row["valid_original"]["ssim"],
                    "pipeline_ssim_valid": row["valid_restored"]["ssim"],
                    "original_temporal_valid": row["valid_original"][
                        "temporal_difference_error"
                    ],
                    "pipeline_temporal_valid": row["valid_restored"][
                        "temporal_difference_error"
                    ],
                }
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Per-sequence CSV: {csv_path}")


if __name__ == "__main__":
    main()
