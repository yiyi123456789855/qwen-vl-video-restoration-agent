import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(project_root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")


def require_dir(path, label):
    if not path.is_dir():
        raise FileNotFoundError(f"{label}不存在：{path}")


def report_matches(report, args, benchmark_id, case_id):
    benchmark = report.get("benchmark") or {}
    return all(
        [
            report.get("fusion_policy") == args.fusion_policy,
            report.get("diagnosis_mode") == args.mode,
            report.get("quality_gate_enabled") is True,
            benchmark.get("benchmark_id") == benchmark_id,
            benchmark.get("case_id") == case_id,
        ]
    )


def select_cases(cases, case_ids, limit, id_only, ood_only):
    selected = list(cases)

    if id_only:
        selected = [case for case in selected if not case["is_ood"]]
    if ood_only:
        selected = [case for case in selected if case["is_ood"]]

    if case_ids:
        requested = set(case_ids)
        known = {case["case_id"] for case in cases}
        unknown = requested - known
        if unknown:
            raise ValueError(
                "manifest中不存在case："
                + ", ".join(sorted(unknown))
            )
        selected = [
            case
            for case in selected
            if case["case_id"] in requested
        ]

    if limit is not None:
        if limit < 1:
            raise ValueError("limit必须大于等于1")
        selected = selected[:limit]

    if not selected:
        raise RuntimeError("没有需要运行的benchmark case")
    return selected


def build_command(
    args,
    pipeline_path,
    qwen_script_path,
    input_dir,
    output_dir,
    denoise_test_script,
    denoise_weights,
    restormer_repo,
    retinexformer_repo,
):
    return [
        sys.executable,
        "-u",
        str(pipeline_path),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--qwen_script",
        str(qwen_script_path),
        "--qwen_model",
        args.qwen_model,
        "--diagnosis_mode",
        args.mode,
        "--diagnosis_frames",
        str(args.diagnosis_frames),
        "--denoise_test_script",
        str(denoise_test_script),
        "--denoise_weights",
        str(denoise_weights),
        "--restormer_repo",
        str(restormer_repo),
        "--retinexformer_repo",
        str(retinexformer_repo),
        "--fusion_policy",
        args.fusion_policy,
        "--cuda_visible_devices",
        args.cuda_visible_devices,
        "--tile",
        str(args.tile),
        "--overlap",
        str(args.overlap),
        "--restormer_tile",
        str(args.restormer_tile),
        "--restormer_overlap",
        str(args.restormer_overlap),
        "--quality_max_attempts",
        str(args.quality_max_attempts),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="benchmarks/heldout_v1.json",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/heldout_closed_loop_v1",
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
        "--denoise_test_script",
        required=True,
    )
    parser.add_argument(
        "--denoise_weights",
        required=True,
    )
    parser.add_argument(
        "--restormer_repo",
        default="third_party/Restormer",
    )
    parser.add_argument(
        "--retinexformer_repo",
        default="third_party/Retinexformer",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "contact_sheet", "native_video"],
        default="single",
    )
    parser.add_argument(
        "--fusion_policy",
        choices=["agreement_only", "objective_override"],
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
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
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
        "--quality_max_attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--case_id",
        action="append",
        default=None,
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--id_only", action="store_true")
    scope.add_argument("--ood_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="单个案例失败后继续运行其余案例",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    pipeline_path = resolve_path(project_root, args.pipeline)
    qwen_script_path = resolve_path(project_root, args.qwen_script)
    denoise_test_script = resolve_path(
        project_root,
        args.denoise_test_script,
    )
    denoise_weights = resolve_path(
        project_root,
        args.denoise_weights,
    )
    restormer_repo = resolve_path(
        project_root,
        args.restormer_repo,
    )
    retinexformer_repo = resolve_path(
        project_root,
        args.retinexformer_repo,
    )

    require_file(manifest_path, "manifest")
    require_file(pipeline_path, "pipeline")
    require_file(qwen_script_path, "Qwen诊断脚本")
    require_file(denoise_test_script, "去噪测试脚本")
    require_file(denoise_weights, "去噪权重")
    require_dir(restormer_repo, "Restormer仓库")
    require_dir(retinexformer_repo, "Retinexformer仓库")

    manifest = load_json(manifest_path)
    benchmark_id = manifest["benchmark_id"]
    cases = select_cases(
        manifest.get("cases", []),
        args.case_id,
        args.limit,
        args.id_only,
        args.ood_only,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    failures = []
    final_statuses = Counter()
    selected_tools = Counter()
    started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        input_dir = resolve_path(project_root, case["input_dir"])
        case_output_dir = output_root / case_id / args.mode
        run_report_path = case_output_dir / "run_report.json"

        print(
            f"\n[{index}/{len(cases)}] {case_id} / {args.mode}",
            flush=True,
        )

        if not input_dir.is_dir():
            failure = {
                "case_id": case_id,
                "error": f"输入目录不存在：{input_dir}",
            }
            failures.append(failure)
            print(f"FAILED: {failure['error']}", flush=True)
            if not args.continue_on_error:
                break
            continue

        if run_report_path.is_file() and not args.rerun:
            existing = load_json(run_report_path)
            if report_matches(
                existing,
                args,
                benchmark_id,
                case_id,
            ):
                print("SKIP: matching closed-loop report exists")
                skipped += 1
                final_statuses[
                    existing.get("closed_loop_status", "unknown")
                ] += 1
                selected_tools[
                    existing.get("selected_tool", "unknown")
                ] += 1
                continue
            print("RERUN: existing report does not match settings")

        command = build_command(
            args=args,
            pipeline_path=pipeline_path,
            qwen_script_path=qwen_script_path,
            input_dir=input_dir,
            output_dir=case_output_dir,
            denoise_test_script=denoise_test_script,
            denoise_weights=denoise_weights,
            restormer_repo=restormer_repo,
            retinexformer_repo=retinexformer_repo,
        )

        try:
            subprocess.run(command, check=True, cwd=project_root)
        except subprocess.CalledProcessError as error:
            failure = {
                "case_id": case_id,
                "returncode": error.returncode,
                "error": "pipeline执行失败",
            }
            failures.append(failure)
            print(
                f"FAILED: returncode={error.returncode}",
                flush=True,
            )
            if not args.continue_on_error:
                break
            continue

        if not run_report_path.is_file():
            failure = {
                "case_id": case_id,
                "error": "pipeline未生成run_report.json",
            }
            failures.append(failure)
            print(f"FAILED: {failure['error']}", flush=True)
            if not args.continue_on_error:
                break
            continue

        report = load_json(run_report_path)
        report["benchmark"] = {
            "benchmark_id": benchmark_id,
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
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        completed += 1
        final_statuses[
            report.get("closed_loop_status", "unknown")
        ] += 1
        selected_tools[
            report.get("selected_tool", "unknown")
        ] += 1

    elapsed = time.perf_counter() - started
    summary = {
        "benchmark_id": benchmark_id,
        "mode": args.mode,
        "fusion_policy": args.fusion_policy,
        "quality_max_attempts": args.quality_max_attempts,
        "requested_cases": len(cases),
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
        "final_statuses": dict(sorted(final_statuses.items())),
        "selected_tools": dict(sorted(selected_tools.items())),
        "elapsed_seconds": round(elapsed, 3),
    }
    summary_path = output_root / f"run_summary_{args.mode}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nClosed-loop benchmark summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {summary_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
