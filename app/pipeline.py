import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.video_denoiser import VideoDenoiser


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


def run_qwen_diagnosis(args, image_path: Path, output_path: Path):
    command = [
        sys.executable,
        "-u",
        str(Path(args.qwen_script).expanduser().resolve()),
        "--model",
        args.qwen_model,
        "--image",
        str(image_path),
        "--output",
        str(output_path),
    ]
    if args.qwen_adapter:
        command.extend(
            [
                "--adapter",
                str(Path(args.qwen_adapter).expanduser().resolve()),
            ]
        )
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(command, check=True, env=environment)


def copy_sequence(frame_paths, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        shutil.copy2(frame_path, target_dir / frame_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--qwen_script", required=True)
    parser.add_argument(
        "--qwen_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--qwen_adapter",
        default=None,
        help="可选；不传时运行Zero-shot基座模型",
    )
    parser.add_argument("--denoise_test_script", required=True)
    parser.add_argument("--denoise_weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入序列不存在：{input_dir}")

    frame_paths = list_images(input_dir)
    if not frame_paths:
        raise RuntimeError(f"输入序列没有图像：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_path = output_dir / "diagnosis.json"
    restored_dir = output_dir / "restored"
    representative = frame_paths[len(frame_paths) // 2]
    started = time.perf_counter()

    print(f"[1/3] Diagnosing representative frame: {representative}")
    run_qwen_diagnosis(
        args,
        representative,
        diagnosis_path,
    )
    diagnosis_report = json.loads(
        diagnosis_path.read_text(encoding="utf-8")
    )
    diagnosis = diagnosis_report["diagnosis"]
    tool = diagnosis.get("recommended_tool", "none")

    denoise_report = None
    if tool == "denoise":
        print("[2/3] Tool selected: denoise")
        denoiser = VideoDenoiser(
            test_script=args.denoise_test_script,
            weights=args.denoise_weights,
            device=args.device,
            tile=args.tile,
            overlap=args.overlap,
        )
        denoise_report = denoiser.run_sequence(
            str(input_dir),
            str(restored_dir),
        )
        action = "denoise"
    else:
        print(f"[2/3] Tool selected: {tool}; copying original frames")
        copy_sequence(frame_paths, restored_dir)
        action = "passthrough" if tool == "none" else "unsupported_passthrough"

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "representative_frame": str(representative),
        "diagnosis": diagnosis,
        "selected_tool": tool,
        "action": action,
        "qwen_adapter": (
            str(Path(args.qwen_adapter).expanduser().resolve())
            if args.qwen_adapter
            else None
        ),
        "denoise": denoise_report,
        "total_runtime_seconds": round(
            time.perf_counter() - started,
            3,
        ),
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[3/3] Completed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
