import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.video_denoiser import VideoDenoiser
from tools.restormer_deblur import RestormerDeblurrer
from tools.retinexformer_lowlight import (
    RetinexformerLowLightEnhancer,
)

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


def select_uniform_frames(
    frame_paths,
    frame_count,
):
    if frame_count < 1:
        raise ValueError(
            "diagnosis_frames必须大于等于1"
        )

    sample_count = min(
        frame_count,
        len(frame_paths),
    )

    if sample_count == 1:
        return [
            frame_paths[len(frame_paths) // 2]
        ]

    last_index = len(frame_paths) - 1

    indices = [
        round(
            index
            * last_index
            / (sample_count - 1)
        )
        for index in range(sample_count)
    ]

    return [
        frame_paths[index]
        for index in indices
    ]


def create_contact_sheet(
    frame_paths,
    output_path,
):
    if not frame_paths:
        raise RuntimeError(
            "无法为零张图像创建拼图"
        )

    cell_width = 320
    cell_height = 180
    columns = min(3, len(frame_paths))
    rows = math.ceil(
        len(frame_paths) / columns
    )

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        color="black",
    )

    for index, frame_path in enumerate(
        frame_paths
    ):
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )

        column = index % columns
        row = index // columns

        x = (
            column * cell_width
            + (cell_width - image.width) // 2
        )
        y = (
            row * cell_height
            + (cell_height - image.height) // 2
        )

        sheet.paste(image, (x, y))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(output_path)

    return output_path


def run_qwen_diagnosis(
    args,
    image_path: Path,
    output_path: Path,
    video_paths=None,
    metric_paths=None,
):
    command = [
        sys.executable,
        "-u",
        str(Path(args.qwen_script).expanduser().resolve()),
        "--model",
        args.qwen_model,
    ]

    if video_paths:
        command.append("--video_frames")
        command.extend(str(path) for path in video_paths)
    else:
        command.extend(
            [
                "--image",
                str(image_path),
            ]
        )

    command.extend(
        [
            "--output",
            str(output_path),
        ]
    )

    if metric_paths:
        command.append("--metric_frames")
        command.extend(
            str(path)
            for path in metric_paths
        )

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
    parser.add_argument(
        "--diagnosis_mode",
        choices=[
            "single",
            "contact_sheet",
            "native_video",
        ],
        default="single",
    )
    parser.add_argument(
        "--diagnosis_frames",
        type=int,
        default=5,
    )
    parser.add_argument("--denoise_test_script", required=True)
    parser.add_argument("--denoise_weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument(
        "--restormer_repo",
        default=None,
    )
    parser.add_argument(
        "--restormer_python",
        default=sys.executable,
    )
    parser.add_argument(
        "--restormer_tile",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--restormer_overlap",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--retinexformer_repo",
        default=None,
    )
    parser.add_argument(
        "--retinexformer_weights",
        default=None,
    )
    parser.add_argument(
        "--retinexformer_config",
        default=None,
    )
    parser.add_argument(
        "--force_tool",
        choices=[
            "none",
            "denoise",
            "deblur",
            "enhance_lowlight",
        ],
        default=None,
        help="仅用于工具链测试",
    )
    parser.add_argument(
        "--diagnosis_only",
        action="store_true",
        help=(
            "只运行退化诊断并保存报告，"
            "不执行任何复原工具"
        ),
    )
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
    sampled_frames = select_uniform_frames(
        frame_paths,
        args.diagnosis_frames,
    )

    contact_sheet_path = None
    video_diagnosis_frames = None
    diagnosis_input = representative

    if args.diagnosis_mode == "contact_sheet":
        contact_sheet_path = (
            output_dir
            / "diagnosis_contact_sheet.png"
        )

        diagnosis_input = create_contact_sheet(
            sampled_frames,
            contact_sheet_path,
        )

    elif args.diagnosis_mode == "native_video":
        video_diagnosis_frames = sampled_frames

    started = time.perf_counter()

    if video_diagnosis_frames is None:
        diagnosis_description = str(diagnosis_input)
    else:
        diagnosis_description = (
            f"{len(video_diagnosis_frames)} ordered frames"
        )

    print(
        "[1/3] Diagnosing "
        f"{args.diagnosis_mode} input: "
        f"{diagnosis_description}"
    )
    run_qwen_diagnosis(
        args,
        diagnosis_input,
        diagnosis_path,
        video_paths=video_diagnosis_frames,
        metric_paths=sampled_frames,
    )
    diagnosis_report = json.loads(
        diagnosis_path.read_text(encoding="utf-8")
    )
    diagnosis = diagnosis_report["diagnosis"]
    raw_diagnosis = diagnosis_report.get(
        "raw_diagnosis",
        diagnosis,
    )
    model_selected_tool = diagnosis.get(
        "recommended_tool",
        "none",
    )
    tool = args.force_tool or model_selected_tool

    denoise_report = None
    deblur_report = None
    lowlight_report = None
    if args.diagnosis_only:
        print(
            "[2/3] Diagnosis-only mode: "
            f"predicted tool={tool}; restoration skipped"
        )
        action = "diagnosis_only"
    elif tool == "denoise":
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
    elif tool == "deblur":
        if not args.restormer_repo:
            raise RuntimeError(
                "模型选择了deblur，"
                "但没有提供--restormer_repo"
            )

        print("[2/3] Tool selected: deblur")

        deblurrer = RestormerDeblurrer(
            repo_dir=args.restormer_repo,
            python_executable=args.restormer_python,
            tile=args.restormer_tile,
            tile_overlap=args.restormer_overlap,
        )

        deblur_report = deblurrer.run_sequence(
            str(input_dir),
            str(restored_dir),
        )

        action = "deblur"
    elif tool == "enhance_lowlight":
        if not args.retinexformer_repo:
            raise RuntimeError(
                "模型选择了enhance_lowlight，"
                "但没有提供--retinexformer_repo"
            )

        print(
            "[2/3] Tool selected: "
            "enhance_lowlight"
        )

        enhancer = RetinexformerLowLightEnhancer(
            repo_dir=args.retinexformer_repo,
            weights=args.retinexformer_weights,
            config=args.retinexformer_config,
            device=args.device,
        )

        lowlight_report = enhancer.run_sequence(
            str(input_dir),
            str(restored_dir),
        )

        action = "enhance_lowlight"
    else:
        print(f"[2/3] Tool selected: {tool}; copying original frames")
        copy_sequence(frame_paths, restored_dir)
        action = "passthrough" if tool == "none" else "unsupported_passthrough"

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "representative_frame": str(representative),
        "diagnosis_mode": args.diagnosis_mode,
        "diagnosis_frames": [
            str(path)
            for path in sampled_frames
        ],
        "diagnosis_input": str(
            diagnosis_input
        ),
        "contact_sheet": (
            None
            if contact_sheet_path is None
            else str(contact_sheet_path)
        ),
        "native_video_frames": (
            None
            if video_diagnosis_frames is None
            else [
                str(path)
                for path in video_diagnosis_frames
            ]
        ),
        "diagnosis": diagnosis,
        "raw_diagnosis": raw_diagnosis,
        "objective_prior": diagnosis_report.get(
            "objective_prior"
        ),
        "decision_source": diagnosis_report.get(
            "decision_source"
        ),
        "routing_confidence": diagnosis.get(
            "confidence"
        ),
        "action": action,
        "qwen_adapter": (
            str(Path(args.qwen_adapter).expanduser().resolve())
            if args.qwen_adapter
            else None
        ),
        "model_selected_tool": model_selected_tool,
        "raw_model_selected_tool": (
            raw_diagnosis.get("recommended_tool")
        ),
        "selected_tool": tool,
        "force_tool": args.force_tool,
        "diagnosis_only": args.diagnosis_only,
        "denoise": denoise_report,
        "deblur": deblur_report,
        "lowlight": lowlight_report,
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
