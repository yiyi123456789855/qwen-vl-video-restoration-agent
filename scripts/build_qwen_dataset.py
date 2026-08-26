import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


PROMPT = """
你是一个无参考图像质量诊断器。输入图像可能完全干净，也可能存在退化。

可选标签：
degradation只能选择：clean、noise、blur、jpeg、low_light、unknown
severity只能选择：none、mild、medium、severe
recommended_tool只能选择：
none、denoise、deblur、dejpeg、enhance_low_light

判断原则：
1. 如果没有明确可见的退化，必须判断为clean、none、none。
2. 只有看到明显随机颗粒、孤立亮暗点或结构被随机噪声破坏时，才判断为noise。
3. 灰度图、黑白色调、辐射图像本身的纹理和较低对比度，不等于存在噪声。
4. 不要因为任务涉及图像复原，就默认图像一定需要处理。
5. 选择最主要的一种退化；证据不足时选择unknown和none。
6. reason用不超过40个汉字说明可见依据。

只输出一个JSON对象，不要输出Markdown。
JSON必须只包含以下四个键：
degradation、severity、recommended_tool、reason。
""".strip()


def choose_frames(paths, count):
    paths = sorted(paths)
    if count <= 0 or count >= len(paths):
        return paths

    if count == 1:
        return [paths[len(paths) // 2]]

    indices = {
        round(i * (len(paths) - 1) / (count - 1))
        for i in range(count)
    }
    return [paths[i] for i in sorted(indices)]


def load_gray(path):
    return (
        np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        / 255.0
    )


def calculate_psnr(noisy_path, clean_path):
    noisy = load_gray(noisy_path)
    clean = load_gray(clean_path)

    if noisy.shape != clean.shape:
        raise ValueError(
            f"尺寸不一致：{noisy_path} {noisy.shape}，"
            f"{clean_path} {clean.shape}"
        )

    mse = float(np.mean((noisy - clean) ** 2))
    if mse <= 1e-12:
        return 99.0

    return 10.0 * math.log10(1.0 / mse)


def psnr_to_severity(psnr):
    if psnr >= 35.0:
        return "mild"
    if psnr >= 25.0:
        return "medium"
    return "severe"


def make_answer(label, severity):
    if label == "clean":
        result = {
            "degradation": "clean",
            "severity": "none",
            "recommended_tool": "none",
            "reason": "未观察到需要处理的明显退化。",
        }
    else:
        reasons = {
            "mild": "存在轻微随机颗粒，建议轻度去噪。",
            "medium": "存在明显随机噪声，建议进行去噪。",
            "severe": "随机噪声严重破坏细节，需要强力去噪。",
        }
        result = {
            "degradation": "noise",
            "severity": severity,
            "recommended_tool": "denoise",
            "reason": reasons[severity],
        }

    return json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_sample(
    image_path,
    label,
    severity,
    split,
    sequence,
    frame_name,
    pair_psnr,
):
    return {
        "image": str(image_path.resolve()),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": make_answer(label, severity),
                    }
                ],
            },
        ],
        "label": label,
        "severity": severity,
        "recommended_tool": (
            "none" if label == "clean" else "denoise"
        ),
        "split": split,
        "sequence": sequence,
        "frame": frame_name,
        "pair_psnr": (
            None if pair_psnr is None else round(pair_psnr, 4)
        ),
    }


def build_split(dataset_root, split, frames_per_sequence):
    noisy_root = dataset_root / split / "noisy"
    clean_root = dataset_root / split / "clean"

    if not noisy_root.is_dir():
        raise FileNotFoundError(f"目录不存在：{noisy_root}")
    if not clean_root.is_dir():
        raise FileNotFoundError(f"目录不存在：{clean_root}")

    samples = []
    missing_pairs = []

    sequence_dirs = sorted(
        path for path in noisy_root.iterdir() if path.is_dir()
    )

    for noisy_sequence in sequence_dirs:
        clean_sequence = clean_root / noisy_sequence.name
        if not clean_sequence.is_dir():
            missing_pairs.append(str(clean_sequence))
            continue

        frame_paths = choose_frames(
            noisy_sequence.glob("*.png"),
            frames_per_sequence,
        )

        for noisy_path in frame_paths:
            clean_path = clean_sequence / noisy_path.name
            if not clean_path.is_file():
                missing_pairs.append(str(clean_path))
                continue

            pair_psnr = calculate_psnr(noisy_path, clean_path)
            severity = psnr_to_severity(pair_psnr)

            samples.append(
                make_sample(
                    image_path=noisy_path,
                    label="noise",
                    severity=severity,
                    split=split,
                    sequence=noisy_sequence.name,
                    frame_name=noisy_path.name,
                    pair_psnr=pair_psnr,
                )
            )

            samples.append(
                make_sample(
                    image_path=clean_path,
                    label="clean",
                    severity="none",
                    split=split,
                    sequence=noisy_sequence.name,
                    frame_name=clean_path.name,
                    pair_psnr=None,
                )
            )

    return samples, missing_pairs


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--frames_per_sequence",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_root": str(dataset_root),
        "frames_per_sequence": args.frames_per_sequence,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        samples, missing_pairs = build_split(
            dataset_root,
            split,
            args.frames_per_sequence,
        )

        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, samples)

        labels = Counter(row["label"] for row in samples)
        severities = Counter(row["severity"] for row in samples)

        summary["splits"][split] = {
            "samples": len(samples),
            "labels": dict(labels),
            "severities": dict(severities),
            "missing_pairs": missing_pairs,
            "output": str(output_path),
        }

        print(
            f"{split}: {len(samples)} samples, "
            f"labels={dict(labels)}, "
            f"severity={dict(severities)}, "
            f"missing={len(missing_pairs)}"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()