import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.objective_prior import analyze_sequence
except ModuleNotFoundError:
    from objective_prior import analyze_sequence


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_THRESHOLDS = {
    "denoise_min_noise_reduction": 0.10,
    "denoise_min_gradient_retention": 0.25,
    "deblur_min_sharpness_gain": 1.10,
    "deblur_max_noise_growth": 2.0,
    "lowlight_min_luminance_gain": 1.20,
    "lowlight_min_dark_area_reduction": 0.10,
    "lowlight_max_highlight_clip_increase": 0.08,
    "max_temporal_residual_ratio": 1.35,
}


def natural_key(path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def list_images(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{folder}")

    paths = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_key,
    )
    if not paths:
        raise RuntimeError(f"图像目录为空：{folder}")
    return paths


def pair_images(input_paths, output_paths):
    input_by_stem = {path.stem: path for path in input_paths}
    output_by_stem = {path.stem: path for path in output_paths}

    if len(input_by_stem) != len(input_paths):
        raise RuntimeError("输入目录存在重复文件stem")
    if len(output_by_stem) != len(output_paths):
        raise RuntimeError("输出目录存在重复文件stem")

    missing = sorted(set(input_by_stem) - set(output_by_stem))
    extra = sorted(set(output_by_stem) - set(input_by_stem))
    if missing or extra:
        raise RuntimeError(
            "输入输出帧不匹配："
            f"missing={missing}, extra={extra}"
        )

    return [
        (input_by_stem[stem], output_by_stem[stem])
        for stem in sorted(
            input_by_stem,
            key=lambda value: natural_key(
                input_by_stem[value]
            ),
        )
    ]


def load_luminance(path):
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def measure_extra_frame(path):
    luminance = load_luminance(path)
    gradient_x = np.diff(luminance, axis=1)
    gradient_y = np.diff(luminance, axis=0)

    return {
        "bright_clip_ratio": float(
            np.mean(luminance >= 0.98)
        ),
        "black_clip_ratio": float(
            np.mean(luminance <= 0.01)
        ),
        "gradient_energy": float(
            (
                np.mean(np.abs(gradient_x))
                + np.mean(np.abs(gradient_y))
            )
            / 2.0
        ),
    }


def aggregate_extra(paths):
    measurements = [
        measure_extra_frame(path)
        for path in paths
    ]
    summary = {}

    for key in measurements[0]:
        values = [item[key] for item in measurements]
        summary[key] = {
            "median": round(float(np.median(values)), 6),
            "min": round(float(np.min(values)), 6),
            "max": round(float(np.max(values)), 6),
        }

    return summary


def temporal_stats(paths):
    frames = [load_luminance(path) for path in paths]
    if len(frames) < 2:
        return {
            "frame_pairs": 0,
            "normalized_residual_median": None,
            "mean_luminance_cv": 0.0,
        }

    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise RuntimeError("序列帧尺寸不一致")

    normalized = [
        frame / max(float(frame.mean()), 1e-6)
        for frame in frames
    ]
    residuals = []

    for previous, current in zip(
        normalized[:-1],
        normalized[1:],
    ):
        difference = current - previous
        centered = difference - np.median(difference)
        residuals.append(
            float(np.median(np.abs(centered)))
        )

    luminance_means = np.asarray(
        [float(frame.mean()) for frame in frames],
        dtype=np.float64,
    )
    coefficient_of_variation = (
        float(luminance_means.std())
        / max(float(luminance_means.mean()), 1e-6)
    )

    return {
        "frame_pairs": len(residuals),
        "normalized_residual_median": round(
            float(np.median(residuals)),
            6,
        ),
        "mean_luminance_cv": round(
            coefficient_of_variation,
            6,
        ),
    }


def median_feature(summary, name):
    return float(summary[name]["median"])


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-8:
        return None
    return numerator / denominator


def add_check(
    checks,
    name,
    actual,
    operator,
    threshold,
    description,
):
    if actual is None or not math.isfinite(float(actual)):
        passed = False
    elif operator == ">=":
        passed = actual >= threshold
    elif operator == "<=":
        passed = actual <= threshold
    else:
        raise ValueError(f"未知比较操作：{operator}")

    checks.append(
        {
            "name": name,
            "passed": passed,
            "actual": (
                None
                if actual is None
                else round(float(actual), 6)
            ),
            "operator": operator,
            "threshold": threshold,
            "description": description,
        }
    )


def evaluate_quality(
    input_dir,
    output_dir,
    tool,
    attempt=1,
    max_attempts=2,
    threshold_overrides=None,
):
    if attempt < 1:
        raise ValueError("attempt必须大于等于1")
    if max_attempts < 1:
        raise ValueError("max_attempts必须大于等于1")

    thresholds = dict(DEFAULT_THRESHOLDS)
    if threshold_overrides:
        unknown_thresholds = (
            set(threshold_overrides)
            - set(DEFAULT_THRESHOLDS)
        )
        if unknown_thresholds:
            raise ValueError(
                "未知质量阈值："
                + ", ".join(sorted(unknown_thresholds))
            )
        thresholds.update(threshold_overrides)

    input_paths = list_images(input_dir)
    output_paths = list_images(output_dir)
    pairs = pair_images(input_paths, output_paths)
    paired_input_paths = [pair[0] for pair in pairs]
    paired_output_paths = [pair[1] for pair in pairs]

    before_objective = analyze_sequence(
        paired_input_paths
    )
    after_objective = analyze_sequence(
        paired_output_paths
    )
    before_features = before_objective["features"]
    after_features = after_objective["features"]
    before_extra = aggregate_extra(paired_input_paths)
    after_extra = aggregate_extra(paired_output_paths)
    before_temporal = temporal_stats(paired_input_paths)
    after_temporal = temporal_stats(paired_output_paths)

    before_noise = median_feature(
        before_features,
        "noise_sigma",
    )
    after_noise = median_feature(
        after_features,
        "noise_sigma",
    )
    before_sharpness = median_feature(
        before_features,
        "laplacian_variance",
    )
    after_sharpness = median_feature(
        after_features,
        "laplacian_variance",
    )
    before_luminance = median_feature(
        before_features,
        "mean_luminance",
    )
    after_luminance = median_feature(
        after_features,
        "mean_luminance",
    )
    before_dark_ratio = median_feature(
        before_features,
        "dark_pixel_ratio",
    )
    after_dark_ratio = median_feature(
        after_features,
        "dark_pixel_ratio",
    )
    before_gradient = median_feature(
        before_extra,
        "gradient_energy",
    )
    after_gradient = median_feature(
        after_extra,
        "gradient_energy",
    )
    before_bright_clip = median_feature(
        before_extra,
        "bright_clip_ratio",
    )
    after_bright_clip = median_feature(
        after_extra,
        "bright_clip_ratio",
    )

    comparisons = {
        "noise_reduction_fraction": (
            safe_ratio(
                before_noise - after_noise,
                before_noise,
            )
        ),
        "noise_growth_ratio": safe_ratio(
            after_noise,
            before_noise,
        ),
        "sharpness_gain_ratio": safe_ratio(
            after_sharpness,
            before_sharpness,
        ),
        "gradient_retention_ratio": safe_ratio(
            after_gradient,
            before_gradient,
        ),
        "luminance_gain_ratio": safe_ratio(
            after_luminance,
            before_luminance,
        ),
        "dark_pixel_ratio_reduction": (
            before_dark_ratio - after_dark_ratio
        ),
        "bright_clip_increase": (
            after_bright_clip - before_bright_clip
        ),
        "temporal_residual_ratio": safe_ratio(
            after_temporal[
                "normalized_residual_median"
            ],
            before_temporal[
                "normalized_residual_median"
            ],
        ),
    }

    checks = []

    if tool == "denoise":
        add_check(
            checks,
            "noise_reduction",
            comparisons["noise_reduction_fraction"],
            ">=",
            thresholds["denoise_min_noise_reduction"],
            "噪声估计至少下降10%",
        )
        add_check(
            checks,
            "gradient_retention",
            comparisons["gradient_retention_ratio"],
            ">=",
            thresholds["denoise_min_gradient_retention"],
            "避免严重过度平滑",
        )
    elif tool == "deblur":
        add_check(
            checks,
            "sharpness_gain",
            comparisons["sharpness_gain_ratio"],
            ">=",
            thresholds["deblur_min_sharpness_gain"],
            "拉普拉斯清晰度至少提升10%",
        )
        add_check(
            checks,
            "noise_growth",
            comparisons["noise_growth_ratio"],
            "<=",
            thresholds["deblur_max_noise_growth"],
            "去模糊后噪声估计不能超过原来的2倍",
        )
    elif tool == "enhance_lowlight":
        add_check(
            checks,
            "luminance_gain",
            comparisons["luminance_gain_ratio"],
            ">=",
            thresholds["lowlight_min_luminance_gain"],
            "平均亮度至少提升20%",
        )
        add_check(
            checks,
            "dark_area_reduction",
            comparisons["dark_pixel_ratio_reduction"],
            ">=",
            thresholds["lowlight_min_dark_area_reduction"],
            "暗像素比例至少下降10个百分点",
        )
        add_check(
            checks,
            "highlight_clipping",
            comparisons["bright_clip_increase"],
            "<=",
            thresholds[
                "lowlight_max_highlight_clip_increase"
            ],
            "新增高光裁剪比例不超过8个百分点",
        )
    else:
        raise ValueError(
            "tool只能是denoise、deblur或enhance_lowlight"
        )

    if before_temporal["frame_pairs"] > 0:
        add_check(
            checks,
            "temporal_consistency",
            comparisons["temporal_residual_ratio"],
            "<=",
            thresholds["max_temporal_residual_ratio"],
            "归一化帧间残差最多增加35%",
        )

    severe_harm_reasons = []
    if after_bright_clip >= 0.30:
        severe_harm_reasons.append(
            "输出高光裁剪比例达到30%"
        )
    if after_luminance <= 0.01:
        severe_harm_reasons.append("输出几乎全黑")
    temporal_ratio = comparisons[
        "temporal_residual_ratio"
    ]
    if temporal_ratio is not None and temporal_ratio > 2.5:
        severe_harm_reasons.append(
            "时序残差超过输入的2.5倍"
        )

    passed_count = sum(check["passed"] for check in checks)
    quality_score = safe_ratio(passed_count, len(checks))

    if severe_harm_reasons:
        status = "stop"
        reason = "；".join(severe_harm_reasons)
    elif passed_count == len(checks):
        status = "accept"
        reason = "所有工具特定质量门控均通过"
    elif attempt < max_attempts:
        status = "retry"
        failed_names = [
            check["name"]
            for check in checks
            if not check["passed"]
        ]
        reason = "以下检查未通过：" + ", ".join(
            failed_names
        )
    else:
        status = "manual_review"
        reason = "达到最大尝试次数且质量门控仍未全部通过"

    return {
        "quality_gate_version": "quality_gate_v1_development",
        "tool": tool,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "thresholds": thresholds,
        "input_dir": str(Path(input_dir).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "paired_frames": len(pairs),
        "status": status,
        "quality_score": (
            None
            if quality_score is None
            else round(float(quality_score), 4)
        ),
        "reason": reason,
        "checks": checks,
        "severe_harm_reasons": severe_harm_reasons,
        "comparisons": {
            key: (
                None
                if value is None
                else round(float(value), 6)
            )
            for key, value in comparisons.items()
        },
        "before": {
            "objective_features": before_features,
            "extra_features": before_extra,
            "temporal": before_temporal,
        },
        "after": {
            "objective_features": after_features,
            "extra_features": after_extra,
            "temporal": after_temporal,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--tool",
        required=True,
        choices=[
            "denoise",
            "deblur",
            "enhance_lowlight",
        ],
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--deblur_min_sharpness_gain",
        type=float,
        default=DEFAULT_THRESHOLDS[
            "deblur_min_sharpness_gain"
        ],
    )
    parser.add_argument(
        "--max_temporal_residual_ratio",
        type=float,
        default=DEFAULT_THRESHOLDS[
            "max_temporal_residual_ratio"
        ],
    )
    parser.add_argument(
        "--report",
        default=None,
    )
    args = parser.parse_args()

    report = evaluate_quality(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tool=args.tool,
        attempt=args.attempt,
        max_attempts=args.max_attempts,
        threshold_overrides={
            "deblur_min_sharpness_gain": (
                args.deblur_min_sharpness_gain
            ),
            "max_temporal_residual_ratio": (
                args.max_temporal_residual_ratio
            ),
        },
    )

    if args.report:
        report_path = (
            Path(args.report)
            .expanduser()
            .resolve()
        )
    else:
        report_path = (
            Path(args.output_dir)
            .expanduser()
            .resolve()
            .parent
            / "quality_report.json"
        )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
