import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


LOW_LIGHT_MEAN_THRESHOLD = 0.15
LOW_LIGHT_DARK_RATIO_THRESHOLD = 0.90
NOISE_SIGMA_THRESHOLD = 5.0
BLUR_LAPLACIAN_THRESHOLD = 80.0
CLEAN_MEAN_THRESHOLD = 0.25
CLEAN_NOISE_SIGMA_MAX = 2.0
CLEAN_LAPLACIAN_MIN = 100.0


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def measure_frame(path):
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        )

    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )

    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    high_pass = (
        gray[:-2, :-2]
        - 2.0 * gray[:-2, 1:-1]
        + gray[:-2, 2:]
        - 2.0 * gray[1:-1, :-2]
        + 4.0 * center
        - 2.0 * gray[1:-1, 2:]
        + gray[2:, :-2]
        - 2.0 * gray[2:, 1:-1]
        + gray[2:, 2:]
    )

    return {
        "mean_luminance": float(np.mean(gray) / 255.0),
        "dark_pixel_ratio": float(
            np.mean(gray < 0.20 * 255.0)
        ),
        "laplacian_variance": float(
            np.var(laplacian)
        ),
        "noise_sigma": float(
            np.median(np.abs(high_pass))
            / (0.6745 * 6.0)
        ),
    }


def aggregate_measurements(measurements):
    if not measurements:
        raise ValueError("至少需要一帧图像")

    summary = {
        "frame_count": len(measurements),
    }

    for key in measurements[0]:
        values = np.asarray(
            [item[key] for item in measurements],
            dtype=np.float64,
        )
        summary[key] = {
            "median": round(float(np.median(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
        }

    return summary


def classify_prior(feature_summary):
    mean_luminance = feature_summary[
        "mean_luminance"
    ]["median"]
    dark_ratio = feature_summary[
        "dark_pixel_ratio"
    ]["median"]
    laplacian_variance = feature_summary[
        "laplacian_variance"
    ]["median"]
    noise_sigma = feature_summary[
        "noise_sigma"
    ]["median"]

    if (
        mean_luminance < LOW_LIGHT_MEAN_THRESHOLD
        and dark_ratio > LOW_LIGHT_DARK_RATIO_THRESHOLD
    ):
        luminance_margin = clamp(
            (
                LOW_LIGHT_MEAN_THRESHOLD
                - mean_luminance
            )
            / LOW_LIGHT_MEAN_THRESHOLD
        )
        dark_margin = clamp(
            (
                dark_ratio
                - LOW_LIGHT_DARK_RATIO_THRESHOLD
            )
            / (
                1.0
                - LOW_LIGHT_DARK_RATIO_THRESHOLD
            )
        )
        confidence = 0.5 + 0.5 * min(
            luminance_margin,
            dark_margin,
        )
        return {
            "degradation": "low_light",
            "severity": (
                "severe"
                if mean_luminance < 0.08
                else "medium"
            ),
            "recommended_tool": "enhance_lowlight",
            "confidence": round(confidence, 4),
            "reason": "亮度与暗像素比例均达到强低照度条件",
        }

    if noise_sigma >= NOISE_SIGMA_THRESHOLD:
        margin = clamp(
            (
                noise_sigma
                - NOISE_SIGMA_THRESHOLD
            )
            / 10.0
        )
        return {
            "degradation": "noise",
            "severity": (
                "severe"
                if noise_sigma >= 25.0
                else "medium"
                if noise_sigma >= 15.0
                else "mild"
            ),
            "recommended_tool": "denoise",
            "confidence": round(0.5 + 0.5 * margin, 4),
            "reason": "高频噪声估计超过强噪声阈值",
        }

    if (
        laplacian_variance < BLUR_LAPLACIAN_THRESHOLD
        and mean_luminance >= LOW_LIGHT_MEAN_THRESHOLD
        and noise_sigma < CLEAN_NOISE_SIGMA_MAX
    ):
        margin = clamp(
            (
                BLUR_LAPLACIAN_THRESHOLD
                - laplacian_variance
            )
            / 60.0
        )
        return {
            "degradation": "blur",
            "severity": (
                "severe"
                if laplacian_variance < 10.0
                else "medium"
                if laplacian_variance < 35.0
                else "mild"
            ),
            "recommended_tool": "deblur",
            "confidence": round(0.5 + 0.5 * margin, 4),
            "reason": "正常亮度下拉普拉斯方差低于模糊阈值",
        }

    if (
        mean_luminance >= CLEAN_MEAN_THRESHOLD
        and noise_sigma < CLEAN_NOISE_SIGMA_MAX
        and laplacian_variance >= CLEAN_LAPLACIAN_MIN
    ):
        luminance_margin = clamp(
            (
                mean_luminance
                - CLEAN_MEAN_THRESHOLD
            )
            / CLEAN_MEAN_THRESHOLD
        )
        noise_margin = clamp(
            (
                CLEAN_NOISE_SIGMA_MAX
                - noise_sigma
            )
            / CLEAN_NOISE_SIGMA_MAX
        )
        sharpness_margin = clamp(
            (
                laplacian_variance
                - CLEAN_LAPLACIAN_MIN
            )
            / CLEAN_LAPLACIAN_MIN
        )
        confidence = 0.5 + 0.5 * min(
            luminance_margin,
            noise_margin,
            sharpness_margin,
        )
        return {
            "degradation": "clean",
            "severity": "none",
            "recommended_tool": "none",
            "confidence": round(confidence, 4),
            "reason": "亮度、噪声与清晰度均处于干净候选范围",
        }

    return {
        "degradation": "unknown",
        "severity": "none",
        "recommended_tool": "manual_review",
        "confidence": 0.0,
        "reason": "客观特征未落入已校准范围",
    }


def analyze_sequence(frame_paths):
    resolved_paths = [
        Path(path).expanduser().resolve()
        for path in frame_paths
    ]

    missing = [
        str(path)
        for path in resolved_paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "图像不存在：" + ", ".join(missing)
        )

    measurements = [
        measure_frame(path)
        for path in resolved_paths
    ]
    feature_summary = aggregate_measurements(
        measurements
    )
    prior = classify_prior(feature_summary)
    prior["confidence_type"] = (
        "heuristic_margin_uncalibrated"
    )

    return {
        "frames": [str(path) for path in resolved_paths],
        "features": feature_summary,
        "objective_prior": prior,
        "thresholds": {
            "low_light_mean_max": LOW_LIGHT_MEAN_THRESHOLD,
            "low_light_dark_ratio_min": (
                LOW_LIGHT_DARK_RATIO_THRESHOLD
            ),
            "noise_sigma_min": NOISE_SIGMA_THRESHOLD,
            "blur_laplacian_max": BLUR_LAPLACIAN_THRESHOLD,
            "clean_mean_min": CLEAN_MEAN_THRESHOLD,
            "clean_noise_sigma_max": CLEAN_NOISE_SIGMA_MAX,
            "clean_laplacian_min": CLEAN_LAPLACIAN_MIN,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    report = analyze_sequence(args.frames)
    text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )

    if args.output:
        output_path = (
            Path(args.output)
            .expanduser()
            .resolve()
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            text + "\n",
            encoding="utf-8",
        )
        print(f"saved: {output_path}")

    print(text)


if __name__ == "__main__":
    main()
