import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(project_root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="benchmarks/heldout_v1.json",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/heldout_v1",
    )
    parser.add_argument(
        "--pipeline",
        default="app/pipeline.py",
    )
    parser.add_argument(
        "--qwen_script",
        default="models/qwen_diagnoser.py",
    )
    parser.add_argument(
        "--qwen_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "contact_sheet",
            "native_video",
        ],
        default="single",
    )
    parser.add_argument(
        "--fusion_policy",
        choices=[
            "agreement_only",
            "objective_override",
        ],
        default="agreement_only",
    )
    parser.add_argument(
        "--diagnosis_frames",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default="0",
    )
    parser.add_argument(
        "--case_id",
        action="append",
        default=None,
        help="只运行指定case；可以重复传入",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="即使已有run_report.json也重新运行",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    manifest_path = resolve_project_path(
        project_root,
        args.manifest,
    )
    output_root = resolve_project_path(
        project_root,
        args.output_root,
    )
    pipeline_path = resolve_project_path(
        project_root,
        args.pipeline,
    )
    qwen_script_path = resolve_project_path(
        project_root,
        args.qwen_script,
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest不存在：{manifest_path}"
        )
    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"pipeline不存在：{pipeline_path}"
        )
    if not qwen_script_path.is_file():
        raise FileNotFoundError(
            f"Qwen诊断脚本不存在：{qwen_script_path}"
        )

    manifest = load_json(manifest_path)
    cases = manifest.get("cases", [])

    if args.case_id:
        requested = set(args.case_id)
        known = {
            case["case_id"]
            for case in cases
        }
        unknown = requested - known
        if unknown:
            raise ValueError(
                "manifest中不存在case："
                + ", ".join(sorted(unknown))
            )
        cases = [
            case
            for case in cases
            if case["case_id"] in requested
        ]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit必须大于等于1")
        cases = cases[: args.limit]

    if not cases:
        raise RuntimeError("没有需要运行的benchmark case")

    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    failures = []
    started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        input_dir = resolve_project_path(
            project_root,
            case["input_dir"],
        )
        case_output_dir = (
            output_root
            / case_id
            / args.mode
        )
        run_report_path = (
            case_output_dir
            / "run_report.json"
        )

        print(
            f"\n[{index}/{len(cases)}] "
            f"{case_id} / {args.mode}",
            flush=True,
        )

        if not input_dir.is_dir():
            failures.append(
                {
                    "case_id": case_id,
                    "error": f"输入目录不存在：{input_dir}",
                }
            )
            print(f"FAILED: {failures[-1]['error']}")
            continue

        if run_report_path.is_file() and not args.rerun:
            existing_report = load_json(run_report_path)
            existing_policy = existing_report.get(
                "fusion_policy",
                "objective_override",
            )
            if existing_policy == args.fusion_policy:
                print(
                    "SKIP: matching run_report.json "
                    "already exists"
                )
                skipped += 1
                continue
            print(
                "RERUN: existing fusion policy is "
                f"{existing_policy}, requested "
                f"{args.fusion_policy}"
            )

        command = [
            sys.executable,
            "-u",
            str(pipeline_path),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(case_output_dir),
            "--qwen_script",
            str(qwen_script_path),
            "--qwen_model",
            args.qwen_model,
            "--diagnosis_mode",
            args.mode,
            "--diagnosis_frames",
            str(args.diagnosis_frames),
            "--denoise_test_script",
            "unused",
            "--denoise_weights",
            "unused",
            "--cuda_visible_devices",
            args.cuda_visible_devices,
            "--diagnosis_only",
            "--fusion_policy",
            args.fusion_policy,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                cwd=project_root,
            )
        except subprocess.CalledProcessError as error:
            failures.append(
                {
                    "case_id": case_id,
                    "returncode": error.returncode,
                    "error": "pipeline执行失败",
                }
            )
            print(
                f"FAILED: returncode={error.returncode}",
                flush=True,
            )
            continue

        if not run_report_path.is_file():
            failures.append(
                {
                    "case_id": case_id,
                    "error": "pipeline未生成run_report.json",
                }
            )
            print(f"FAILED: {failures[-1]['error']}")
            continue

        run_report = load_json(run_report_path)
        run_report["benchmark"] = {
            "benchmark_id": manifest[
                "benchmark_id"
            ],
            "manifest": str(manifest_path),
            "case_id": case_id,
            "scene_id": case["scene_id"],
            "split": case["split"],
            "generation": case["generation"],
            "expected_degradation": case[
                "expected_degradation"
            ],
            "expected_tool": case["expected_tool"],
            "is_ood": case["is_ood"],
        }
        run_report_path.write_text(
            json.dumps(
                run_report,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completed += 1

    elapsed = time.perf_counter() - started
    summary = {
        "benchmark_id": manifest["benchmark_id"],
        "mode": args.mode,
        "fusion_policy": args.fusion_policy,
        "requested_cases": len(cases),
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": round(elapsed, 3),
    }
    summary_path = output_root / (
        f"run_summary_{args.mode}.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nBenchmark run summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {summary_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
