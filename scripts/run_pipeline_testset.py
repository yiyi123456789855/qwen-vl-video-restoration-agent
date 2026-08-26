import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def image_count(folder: Path) -> int:
    suffixes = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
    }
    if not folder.is_dir():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in suffixes
        for path in folder.iterdir()
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
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
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["noisy", "clean"],
        default=["noisy", "clean"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    test_root = dataset_root / "test"

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    jobs = []
    for mode in args.modes:
        mode_root = test_root / mode
        if not mode_root.is_dir():
            raise FileNotFoundError(f"测试目录不存在：{mode_root}")
        for sequence_dir in sorted(
            path for path in mode_root.iterdir() if path.is_dir()
        ):
            jobs.append((mode, sequence_dir))

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    total_started = time.perf_counter()

    for index, (mode, sequence_dir) in enumerate(jobs, start=1):
        sequence_output = output_root / mode / sequence_dir.name
        report_path = sequence_output / "run_report.json"
        restored_dir = sequence_output / "restored"
        expected_frames = image_count(sequence_dir)

        if (
            not args.force
            and report_path.is_file()
            and image_count(restored_dir) == expected_frames
        ):
            print(
                f"[{index}/{len(jobs)}] SKIP {mode}/{sequence_dir.name}",
                flush=True,
            )
            results.append(
                {
                    "mode": mode,
                    "sequence": sequence_dir.name,
                    "status": "skipped_complete",
                    "output": str(sequence_output),
                }
            )
            continue

        print(
            f"[{index}/{len(jobs)}] RUN {mode}/{sequence_dir.name}",
            flush=True,
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "app.pipeline",
            "--input_dir",
            str(sequence_dir),
            "--output_dir",
            str(sequence_output),
            "--qwen_script",
            str(Path(args.qwen_script).expanduser().resolve()),
            "--qwen_model",
            args.qwen_model,
            "--denoise_test_script",
            str(Path(args.denoise_test_script).expanduser().resolve()),
            "--denoise_weights",
            str(Path(args.denoise_weights).expanduser().resolve()),
            "--device",
            args.device,
            "--cuda_visible_devices",
            args.cuda_visible_devices,
        ]
        if args.qwen_adapter:
            command.extend(
                [
                    "--qwen_adapter",
                    str(Path(args.qwen_adapter).expanduser().resolve()),
                ]
            )

        started = time.perf_counter()
        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
            )
            status = "completed"
            error = None
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = f"exit_code={exc.returncode}"

        results.append(
            {
                "mode": mode,
                "sequence": sequence_dir.name,
                "status": status,
                "error": error,
                "runtime_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
                "output": str(sequence_output),
            }
        )

        summary_path = output_root / "batch_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "dataset_root": str(dataset_root),
                    "output_root": str(output_root),
                    "jobs": len(jobs),
                    "processed": len(results),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    completed = sum(
        row["status"] in {"completed", "skipped_complete"}
        for row in results
    )
    failed = sum(row["status"] == "failed" for row in results)
    final_summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "jobs": len(jobs),
        "completed_or_skipped": completed,
        "failed": failed,
        "total_runtime_seconds": round(
            time.perf_counter() - total_started,
            3,
        ),
        "results": results,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
