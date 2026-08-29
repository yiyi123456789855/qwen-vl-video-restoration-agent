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
from tools.quality_evaluator import evaluate_quality

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
            "--fusion_policy",
            args.fusion_policy,
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


def scaled_overlap(base_overlap, tile, attempt):
    """Increase overlap on retries without reaching the tile size."""
    value = base_overlap * (2 ** (attempt - 1))
    return min(value, max(tile - 1, 0))


def execute_restoration(
    args,
    tool,
    input_dir: Path,
    target_dir: Path,
    attempt,
):
    """Execute one restoration attempt and record its parameters."""
    if tool == "denoise":
        overlap = scaled_overlap(
            args.overlap,
            args.tile,
            attempt,
        )
        parameters = {
            "tile": args.tile,
            "overlap": overlap,
        }
        print(
            "[Restore] "
            f"attempt={attempt}; tool=denoise; "
            f"tile={args.tile}; overlap={overlap}"
        )
        runner = VideoDenoiser(
            test_script=args.denoise_test_script,
            weights=args.denoise_weights,
            device=args.device,
            tile=args.tile,
            overlap=overlap,
        )
        tool_report = runner.run_sequence(
            str(input_dir),
            str(target_dir),
        )
    elif tool == "deblur":
        if not args.restormer_repo:
            raise RuntimeError(
                "模型选择了deblur，"
                "但没有提供--restormer_repo"
            )
        overlap = scaled_overlap(
            args.restormer_overlap,
            args.restormer_tile,
            attempt,
        )
        parameters = {
            "tile": args.restormer_tile,
            "tile_overlap": overlap,
        }
        print(
            "[Restore] "
            f"attempt={attempt}; tool=deblur; "
            f"tile={args.restormer_tile}; "
            f"tile_overlap={overlap}"
        )
        runner = RestormerDeblurrer(
            repo_dir=args.restormer_repo,
            python_executable=args.restormer_python,
            tile=args.restormer_tile,
            tile_overlap=overlap,
        )
        tool_report = runner.run_sequence(
            str(input_dir),
            str(target_dir),
        )
    elif tool == "enhance_lowlight":
        if not args.retinexformer_repo:
            raise RuntimeError(
                "模型选择了enhance_lowlight，"
                "但没有提供--retinexformer_repo"
            )
        parameters = {
            "weights": args.retinexformer_weights,
            "config": args.retinexformer_config,
            "device": args.device,
        }
        print(
            "[Restore] "
            f"attempt={attempt}; "
            "tool=enhance_lowlight"
        )
        runner = RetinexformerLowLightEnhancer(
            repo_dir=args.retinexformer_repo,
            weights=args.retinexformer_weights,
            config=args.retinexformer_config,
            device=args.device,
        )
        tool_report = runner.run_sequence(
            str(input_dir),
            str(target_dir),
        )
    else:
        raise ValueError(f"工具不支持复原执行：{tool}")

    return {
        "attempt": attempt,
        "output_dir": str(target_dir),
        "parameters": parameters,
        "tool_report": tool_report,
    }


def publish_attempt(attempt_dir: Path, restored_dir: Path):
    """Copy the selected attempt to the stable restored/ directory."""
    selected_frames = list_images(attempt_dir)
    if not selected_frames:
        raise RuntimeError(
            f"选中的复原尝试没有输出图像：{attempt_dir}"
        )
    copy_sequence(selected_frames, restored_dir)


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
    parser.add_argument(
        "--fusion_policy",
        choices=[
            "agreement_only",
            "objective_override",
        ],
        default="agreement_only",
        help="路由融合策略；默认使用安全优先的一致性策略",
    )
    parser.add_argument(
        "--disable_quality_gate",
        action="store_true",
        help="关闭工具执行后的客观质量门控",
    )
    parser.add_argument(
        "--quality_attempt",
        type=int,
        default=1,
        help="质量门控的起始尝试编号；通常保持为1",
    )
    parser.add_argument(
        "--quality_max_attempts",
        type=int,
        default=2,
        help="包含首次执行在内的最大复原次数",
    )
    parser.add_argument(
        "--quality_deblur_min_sharpness_gain",
        type=float,
        default=1.10,
        help=(
            "去模糊清晰度提升门槛；默认1.10。"
            "调高可用于门控压力测试"
        ),
    )
    args = parser.parse_args()

    if args.quality_attempt < 1:
        raise ValueError("quality_attempt必须大于等于1")
    if args.quality_max_attempts < args.quality_attempt:
        raise ValueError(
            "quality_max_attempts不能小于quality_attempt"
        )

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
    quality_report = None
    quality_report_path = None
    restoration_attempts = []
    selected_attempt = None
    published_attempt = None
    published_result = None

    if args.diagnosis_only:
        print(
            "[2/3] Diagnosis-only mode: "
            f"predicted tool={tool}; restoration skipped"
        )
        action = "diagnosis_only"
        closed_loop_status = "diagnosis_only"
    elif tool in {
        "denoise",
        "deblur",
        "enhance_lowlight",
    }:
        print(f"[2/3] Tool selected: {tool}")
        action = tool

        quality_gate_enabled = not args.disable_quality_gate
        retry_supported = tool in {"denoise", "deblur"}
        last_attempt = (
            args.quality_max_attempts
            if quality_gate_enabled and retry_supported
            else args.quality_attempt
        )

        for attempt in range(
            args.quality_attempt,
            last_attempt + 1,
        ):
            attempt_dir = (
                output_dir
                / f"restored_attempt_{attempt}"
            )
            attempt_record = execute_restoration(
                args=args,
                tool=tool,
                input_dir=input_dir,
                target_dir=attempt_dir,
                attempt=attempt,
            )

            current_tool_report = attempt_record[
                "tool_report"
            ]
            if tool == "denoise":
                denoise_report = current_tool_report
            elif tool == "deblur":
                deblur_report = current_tool_report
            else:
                lowlight_report = current_tool_report

            if not quality_gate_enabled:
                selected_attempt = attempt
                closed_loop_status = "not_evaluated"
                attempt_record["quality"] = None
                restoration_attempts.append(attempt_record)
                break

            print(
                "[Quality] Evaluating restored sequence: "
                f"tool={tool}; attempt={attempt}"
            )
            quality_report = evaluate_quality(
                input_dir=input_dir,
                output_dir=attempt_dir,
                tool=tool,
                attempt=attempt,
                max_attempts=(
                    args.quality_max_attempts
                    if retry_supported
                    else attempt
                ),
                threshold_overrides={
                    "deblur_min_sharpness_gain": (
                        args.quality_deblur_min_sharpness_gain
                    ),
                },
            )
            attempt_quality_path = (
                output_dir
                / f"quality_attempt_{attempt}.json"
            )
            attempt_quality_path.write_text(
                json.dumps(
                    quality_report,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            attempt_record["quality_report_path"] = str(
                attempt_quality_path
            )
            attempt_record["quality"] = quality_report
            restoration_attempts.append(attempt_record)

            current_status = quality_report["status"]
            print(
                "[Quality] "
                f"attempt={attempt}; "
                f"status={current_status}; "
                f"score={quality_report['quality_score']}"
            )

            if current_status in {"accept", "stop"}:
                selected_attempt = attempt
                closed_loop_status = current_status
                break

            if current_status == "retry" and retry_supported:
                print(
                    "[Closed loop] Quality gate requested retry; "
                    "increasing tile overlap"
                )
                continue

            selected_attempt = attempt
            closed_loop_status = "manual_review"
            if current_status == "retry":
                quality_report = dict(quality_report)
                quality_report["status"] = "manual_review"
                quality_report["reason"] = (
                    "质量门控建议重试，但该工具没有安全且不同的"
                    "自动重试参数，转人工复核"
                )
                restoration_attempts[-1][
                    "quality"
                ] = quality_report
            break

        if selected_attempt is None:
            selected_attempt = restoration_attempts[-1][
                "attempt"
            ]

        if closed_loop_status == "manual_review":
            selected_attempt = max(
                restoration_attempts,
                key=lambda record: (
                    -1.0
                    if record.get("quality") is None
                    or record["quality"].get(
                        "quality_score"
                    ) is None
                    else record["quality"][
                        "quality_score"
                    ],
                    record["attempt"],
                ),
            )["attempt"]

        selected_record = next(
            record
            for record in restoration_attempts
            if record["attempt"] == selected_attempt
        )
        if closed_loop_status == "stop":
            copy_sequence(frame_paths, restored_dir)
            published_result = "original_input_safety_fallback"
        else:
            publish_attempt(
                Path(selected_record["output_dir"]),
                restored_dir,
            )
            published_attempt = selected_attempt
            published_result = "restoration_attempt"

        quality_report_path = (
            None
            if quality_report is None
            else output_dir / "quality_report.json"
        )
        if quality_report_path is not None:
            quality_report_path.write_text(
                json.dumps(
                    quality_report,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    else:
        print(f"[2/3] Tool selected: {tool}; copying original frames")
        copy_sequence(frame_paths, restored_dir)
        action = "passthrough" if tool == "none" else "unsupported_passthrough"
        if tool == "none":
            closed_loop_status = "accept"
        elif tool == "manual_review":
            closed_loop_status = "manual_review"
        else:
            closed_loop_status = "not_evaluated"

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
        "fusion_policy": diagnosis_report.get(
            "fusion_policy",
            args.fusion_policy,
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
        "quality_gate_enabled": (
            not args.disable_quality_gate
        ),
        "closed_loop_status": closed_loop_status,
        "quality_report_path": (
            None
            if quality_report_path is None
            else str(quality_report_path)
        ),
        "quality": quality_report,
        "selected_attempt": selected_attempt,
        "published_attempt": published_attempt,
        "published_result": published_result,
        "restoration_attempts": restoration_attempts,
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
