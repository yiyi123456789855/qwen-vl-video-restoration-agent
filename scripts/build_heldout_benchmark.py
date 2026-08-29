import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SCENES = [
    "GOPR0385_11_01",
    "GOPR0854_11_00",
    "GOPR0881_11_01",
]

DEFAULT_EXCLUDED_SCENES = [
    "GOPR0384_11_00",
    "GOPR0384_11_05",
]


CASE_SPECS = [
    {
        "name": "clean",
        "expected_degradation": "clean",
        "expected_tool": "none",
        "is_ood": False,
        "parameters": {},
    },
    {
        "name": "blur",
        "expected_degradation": "blur",
        "expected_tool": "deblur",
        "is_ood": False,
        "parameters": {"source": "GoPro input"},
    },
    {
        "name": "noise_sigma12",
        "expected_degradation": "noise",
        "expected_tool": "denoise",
        "is_ood": False,
        "parameters": {"sigma": 12.0},
    },
    {
        "name": "noise_sigma35",
        "expected_degradation": "noise",
        "expected_tool": "denoise",
        "is_ood": False,
        "parameters": {"sigma": 35.0},
    },
    {
        "name": "lowlight_gain030",
        "expected_degradation": "low_light",
        "expected_tool": "enhance_lowlight",
        "is_ood": False,
        "parameters": {"gain": 0.30},
    },
    {
        "name": "lowlight_gain012",
        "expected_degradation": "low_light",
        "expected_tool": "enhance_lowlight",
        "is_ood": False,
        "parameters": {"gain": 0.12},
    },
    {
        "name": "jpeg_quality10",
        "expected_degradation": "jpeg",
        "expected_tool": "manual_review",
        "is_ood": True,
        "parameters": {"quality": 10},
    },
    {
        "name": "mixed_blur_noise",
        "expected_degradation": "mixed",
        "expected_tool": "manual_review",
        "is_ood": True,
        "parameters": {
            "source": "GoPro input",
            "sigma": 18.0,
        },
    },
    {
        "name": "unknown_color_cast",
        "expected_degradation": "unknown",
        "expected_tool": "manual_review",
        "is_ood": True,
        "parameters": {
            "rgb_scale": [0.45, 1.0, 0.45],
        },
    },
]


def select_uniform(items, count):
    if count < 1:
        raise ValueError("frame_count必须大于等于1")
    if len(items) < count:
        raise RuntimeError(
            f"可用帧不足：需要{count}，实际{len(items)}"
        )
    if count == 1:
        return [items[len(items) // 2]]

    last_index = len(items) - 1
    indices = [
        round(index * last_index / (count - 1))
        for index in range(count)
    ]
    return [items[index] for index in indices]


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


def add_gaussian_noise(array, sigma, rng):
    noise = rng.normal(
        0.0,
        sigma,
        array.shape,
    )
    return np.clip(
        array.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)


def transform_case(
    case_name,
    clean,
    blur,
    parameters,
    rng,
):
    if case_name == "clean":
        return clean, ".png", {}

    if case_name == "blur":
        return blur, ".png", {}

    if case_name.startswith("noise_sigma"):
        result = add_gaussian_noise(
            clean,
            parameters["sigma"],
            rng,
        )
        return result, ".png", {}

    if case_name.startswith("lowlight_gain"):
        result = np.clip(
            clean.astype(np.float32)
            * parameters["gain"],
            0,
            255,
        ).astype(np.uint8)
        return result, ".png", {}

    if case_name == "jpeg_quality10":
        return (
            clean,
            ".jpg",
            {
                "quality": parameters["quality"],
                "subsampling": 2,
            },
        )

    if case_name == "mixed_blur_noise":
        result = add_gaussian_noise(
            blur,
            parameters["sigma"],
            rng,
        )
        return result, ".png", {}

    if case_name == "unknown_color_cast":
        scale = np.asarray(
            parameters["rgb_scale"],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        result = np.clip(
            clean.astype(np.float32) * scale,
            0,
            255,
        ).astype(np.uint8)
        return result, ".png", {}

    raise ValueError(f"未知case：{case_name}")


def collect_scene_pairs(
    input_dir,
    target_dir,
    scene_id,
):
    pairs = []
    pattern = f"{scene_id}-*.png"

    for input_path in sorted(input_dir.glob(pattern)):
        target_path = target_dir / input_path.name
        if target_path.is_file():
            pairs.append((input_path, target_path))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gopro_root",
        default=(
            "third_party/Restormer/Motion_Deblurring/"
            "Datasets/test/GoPro"
        ),
    )
    parser.add_argument(
        "--output_root",
        default="samples/heldout_v1",
    )
    parser.add_argument(
        "--manifest",
        default="benchmarks/heldout_v1.json",
    )
    parser.add_argument(
        "--benchmark_id",
        default="gopro_heldout_v1",
    )
    parser.add_argument(
        "--split",
        default="heldout",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
    )
    parser.add_argument(
        "--frame_count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4090,
    )
    parser.add_argument(
        "--exclude_scene",
        action="append",
        default=list(DEFAULT_EXCLUDED_SCENES),
        help=(
            "禁止用于当前benchmark的场景；可重复传入。"
            "默认排除GOPR0384开发场景"
        ),
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    gopro_root = (
        Path(args.gopro_root)
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(args.output_root)
        .expanduser()
        .resolve()
    )
    manifest_path = (
        Path(args.manifest)
        .expanduser()
        .resolve()
    )

    input_dir = gopro_root / "input"
    target_dir = gopro_root / "target"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"GoPro input不存在：{input_dir}"
        )
    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"GoPro target不存在：{target_dir}"
        )

    cases = []

    excluded_scenes = set(args.exclude_scene)
    selected_exclusions = set(args.scenes) & excluded_scenes
    if selected_exclusions:
        raise RuntimeError(
            "benchmark场景与排除列表冲突："
            + ", ".join(sorted(selected_exclusions))
        )

    for scene_index, scene_id in enumerate(args.scenes):
        if scene_id.startswith("GOPR0384"):
            raise RuntimeError(
                "held-out场景不能使用开发集GOPR0384"
            )

        pairs = collect_scene_pairs(
            input_dir,
            target_dir,
            scene_id,
        )
        selected_pairs = select_uniform(
            pairs,
            args.frame_count,
        )

        for case_index, spec in enumerate(CASE_SPECS):
            case_id = f"{scene_id}__{spec['name']}"
            case_dir = output_root / case_id
            case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            rng = np.random.default_rng(
                args.seed
                + scene_index * 1000
                + case_index * 100
            )
            output_names = []
            source_names = []

            for frame_index, (
                input_path,
                target_path,
            ) in enumerate(selected_pairs, start=1):
                clean = load_rgb(target_path)
                blur = load_rgb(input_path)
                result, suffix, save_kwargs = transform_case(
                    spec["name"],
                    clean,
                    blur,
                    spec["parameters"],
                    rng,
                )
                output_name = f"{frame_index:04d}{suffix}"
                output_path = case_dir / output_name
                Image.fromarray(result).save(
                    output_path,
                    **save_kwargs,
                )
                output_names.append(output_name)
                source_names.append(target_path.name)

            try:
                relative_input_dir = case_dir.relative_to(
                    project_root
                )
            except ValueError:
                relative_input_dir = case_dir

            cases.append(
                {
                    "case_id": case_id,
                    "scene_id": scene_id,
                    "split": args.split,
                    "input_dir": str(relative_input_dir),
                    "frame_count": len(output_names),
                    "frame_files": output_names,
                    "source_frame_files": source_names,
                    "expected_degradation": spec[
                        "expected_degradation"
                    ],
                    "expected_tool": spec[
                        "expected_tool"
                    ],
                    "is_ood": spec["is_ood"],
                    "generation": {
                        "name": spec["name"],
                        **spec["parameters"],
                    },
                }
            )

    manifest = {
        "benchmark_id": args.benchmark_id,
        "split": args.split,
        "seed": args.seed,
        "frame_count_per_case": args.frame_count,
        "development_scene_exclusions": sorted(
            excluded_scenes
        ),
        "scenes": args.scenes,
        "case_count": len(cases),
        "cases": cases,
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"scenes: {len(args.scenes)}")
    print(f"cases: {len(cases)}")
    print(
        "images: "
        f"{len(cases) * args.frame_count}"
    )
    print(f"output root: {output_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
