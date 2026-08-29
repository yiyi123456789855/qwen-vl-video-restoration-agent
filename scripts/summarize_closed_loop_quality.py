import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


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

RESTORATION_TOOLS = {
    "denoise",
    "deblur",
    "enhance_lowlight",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def values_differ(left, right):
    try:
        return not math.isclose(
            float(left),
            float(right),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return left != right


def uses_nondefault_thresholds(quality):
    thresholds = quality.get("thresholds") or {}
    for name, default_value in DEFAULT_THRESHOLDS.items():
        if name in thresholds and values_differ(
            thresholds[name],
            default_value,
        ):
            return True
    return False


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values):
    numeric = [
        value
        for value in (safe_float(item) for item in values)
        if value is not None
    ]
    if not numeric:
        return None
    return statistics.fmean(numeric)


def format_number(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_percent(value):
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def markdown_escape(value):
    return str(value).replace("|", "\\|")


def attempt_runtime(attempt):
    tool_report = attempt.get("tool_report") or {}
    return safe_float(tool_report.get("runtime_seconds"))


def attempt_peak_gpu(attempt):
    tool_report = attempt.get("tool_report") or {}
    for key in (
        "peak_gpu_memory_gb",
        "peak_memory_gb",
        "peak_gpu_gb",
    ):
        value = safe_float(tool_report.get(key))
        if value is not None:
            return value
    return None


def flatten_report(path, report):
    quality = report.get("quality")
    if not isinstance(quality, dict):
        return None

    tool = report.get("selected_tool") or quality.get("tool")
    if tool not in RESTORATION_TOOLS:
        return None

    attempts = report.get("restoration_attempts") or []
    if attempts:
        attempt_count = len(attempts)
        attempt_statuses = [
            (attempt.get("quality") or {}).get(
                "status",
                "not_evaluated",
            )
            for attempt in attempts
        ]
        tool_runtime = sum(
            value
            for value in (
                attempt_runtime(attempt)
                for attempt in attempts
            )
            if value is not None
        )
        peak_values = [
            value
            for value in (
                attempt_peak_gpu(attempt)
                for attempt in attempts
            )
            if value is not None
        ]
        peak_gpu = max(peak_values) if peak_values else None
    else:
        attempt_count = int(quality.get("attempt", 1))
        attempt_statuses = [quality.get("status", "unknown")]
        legacy_tool_report = (
            report.get("denoise")
            or report.get("deblur")
            or report.get("lowlight")
            or {}
        )
        tool_runtime = safe_float(
            legacy_tool_report.get("runtime_seconds")
        )
        peak_gpu = safe_float(
            legacy_tool_report.get("peak_gpu_memory_gb")
        )

    forced = report.get("force_tool") is not None
    stress_test = uses_nondefault_thresholds(quality)
    benchmark = report.get("benchmark") or {}
    status = report.get("closed_loop_status") or quality.get(
        "status",
        "unknown",
    )

    return {
        "case": benchmark.get("case_id", path.parent.name),
        "report_path": str(path),
        "tool": tool,
        "expected_tool": benchmark.get("expected_tool"),
        "is_ood": benchmark.get("is_ood"),
        "degradation": (
            report.get("diagnosis") or {}
        ).get("degradation", "unknown"),
        "status": status,
        "attempt_count": attempt_count,
        "retry_triggered": attempt_count > 1,
        "attempt_statuses": " -> ".join(attempt_statuses),
        "selected_attempt": report.get("selected_attempt"),
        "published_attempt": report.get("published_attempt"),
        "quality_score": safe_float(quality.get("quality_score")),
        "tool_runtime_seconds": tool_runtime,
        "total_runtime_seconds": safe_float(
            report.get("total_runtime_seconds")
        ),
        "peak_gpu_memory_gb": peak_gpu,
        "routing_confidence": safe_float(
            report.get("routing_confidence")
        ),
        "decision_source": report.get("decision_source"),
        "forced": forced,
        "stress_test": stress_test,
    }


def discover_reports(outputs_dir, explicit_reports):
    if explicit_reports:
        paths = [
            Path(value).expanduser().resolve()
            for value in explicit_reports
        ]
    else:
        root = Path(outputs_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"输出目录不存在：{root}")
        paths = sorted(root.rglob("run_report.json"))

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "以下运行报告不存在：" + ", ".join(missing)
        )
    return paths


def aggregate_rows(rows, label):
    count = len(rows)
    if count == 0:
        return None

    statuses = Counter(row["status"] for row in rows)
    return {
        "tool": label,
        "runs": count,
        "accept_rate": statuses["accept"] / count,
        "retry_rate": sum(
            row["retry_triggered"] for row in rows
        ) / count,
        "manual_review_rate": (
            statuses["manual_review"] / count
        ),
        "stop_rate": statuses["stop"] / count,
        "mean_attempts": mean_or_none(
            row["attempt_count"] for row in rows
        ),
        "mean_quality_score": mean_or_none(
            row["quality_score"] for row in rows
        ),
        "mean_tool_runtime_seconds": mean_or_none(
            row["tool_runtime_seconds"] for row in rows
        ),
        "mean_total_runtime_seconds": mean_or_none(
            row["total_runtime_seconds"] for row in rows
        ),
        "mean_peak_gpu_memory_gb": mean_or_none(
            row["peak_gpu_memory_gb"] for row in rows
        ),
    }


def build_aggregates(rows):
    aggregates = [aggregate_rows(rows, "all")]
    for tool in sorted({row["tool"] for row in rows}):
        tool_rows = [row for row in rows if row["tool"] == tool]
        aggregates.append(aggregate_rows(tool_rows, tool))
    return [item for item in aggregates if item is not None]


def write_csv(path, rows):
    fieldnames = [
        "case",
        "tool",
        "expected_tool",
        "is_ood",
        "degradation",
        "status",
        "attempt_count",
        "retry_triggered",
        "attempt_statuses",
        "selected_attempt",
        "published_attempt",
        "quality_score",
        "tool_runtime_seconds",
        "total_runtime_seconds",
        "peak_gpu_memory_gb",
        "routing_confidence",
        "decision_source",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def markdown_table(headers, rows, alignments=None):
    if alignments is None:
        alignments = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_escape(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def build_markdown(rows, aggregates, excluded):
    per_run_rows = []
    for row in rows:
        per_run_rows.append(
            [
                row["case"],
                row["tool"],
                row["expected_tool"] or "-",
                (
                    "yes"
                    if row["is_ood"] is True
                    else "no"
                    if row["is_ood"] is False
                    else "-"
                ),
                row["degradation"],
                row["status"],
                row["attempt_count"],
                row["attempt_statuses"],
                format_number(row["quality_score"], 4),
                format_number(row["tool_runtime_seconds"]),
                format_number(row["total_runtime_seconds"]),
                format_number(row["peak_gpu_memory_gb"]),
            ]
        )

    aggregate_rows = []
    for row in aggregates:
        aggregate_rows.append(
            [
                row["tool"],
                row["runs"],
                format_percent(row["accept_rate"]),
                format_percent(row["retry_rate"]),
                format_percent(row["manual_review_rate"]),
                format_percent(row["stop_rate"]),
                format_number(row["mean_attempts"]),
                format_number(row["mean_quality_score"], 4),
                format_number(row["mean_tool_runtime_seconds"]),
                format_number(row["mean_total_runtime_seconds"]),
                format_number(row["mean_peak_gpu_memory_gb"]),
            ]
        )

    parts = [
        "# Closed-Loop Restoration Quality Report",
        "",
        (
            f"Included runs: {len(rows)}. "
            f"Excluded forced/stress runs: {excluded}."
        ),
        "",
        "## Per-run results",
        "",
        markdown_table(
            [
                "Case",
                "Tool",
                "Expected",
                "OOD",
                "Degradation",
                "Final status",
                "Attempts",
                "Attempt states",
                "Quality",
                "Tool s",
                "Total s",
                "Peak GB",
            ],
            per_run_rows,
            [
                "---",
                "---",
                "---",
                "---",
                "---",
                "---",
                "---:",
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
            ],
        ),
        "",
        "## Aggregate by tool",
        "",
        markdown_table(
            [
                "Tool",
                "Runs",
                "Accept",
                "Retry",
                "Manual review",
                "Stop",
                "Mean attempts",
                "Mean quality",
                "Mean tool s",
                "Mean total s",
                "Mean peak GB",
            ],
            aggregate_rows,
            [
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ],
        ),
        "",
        "## Metric definitions",
        "",
        "- Accept rate: final closed-loop status is `accept`.",
        "- Retry rate: more than one restoration attempt was executed.",
        "- Manual review rate: final status is `manual_review`.",
        "- Stop rate: severe quality harm triggered the safety stop.",
        "- Pressure tests and forced-tool runs are excluded by default.",
        "",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs_dir",
        default="outputs",
    )
    parser.add_argument(
        "--result_dir",
        required=True,
    )
    parser.add_argument(
        "--run_report",
        action="append",
        default=None,
        help=(
            "显式指定run_report.json；可重复传入。"
            "不传时递归扫描outputs_dir"
        ),
    )
    parser.add_argument(
        "--include_forced",
        action="store_true",
    )
    parser.add_argument(
        "--include_stress_tests",
        action="store_true",
    )
    args = parser.parse_args()

    report_paths = discover_reports(
        args.outputs_dir,
        args.run_report,
    )
    candidates = []
    unreadable = []

    for path in report_paths:
        try:
            row = flatten_report(path, read_json(path))
        except (OSError, ValueError, TypeError) as error:
            unreadable.append((str(path), str(error)))
            continue
        if row is not None:
            candidates.append(row)

    rows = []
    excluded = 0
    for row in candidates:
        if row["forced"] and not args.include_forced:
            excluded += 1
            continue
        if row["stress_test"] and not args.include_stress_tests:
            excluded += 1
            continue
        rows.append(row)

    rows.sort(key=lambda row: (row["tool"], row["case"]))
    if not rows:
        raise RuntimeError(
            "没有可汇总的默认阈值、非强制闭环运行。"
            "请先运行至少一个完整复原案例，或使用include选项。"
        )

    aggregates = build_aggregates(rows)
    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / "closed_loop_quality.csv"
    json_path = result_dir / "closed_loop_quality.json"
    markdown_path = result_dir / "closed_loop_quality.md"

    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "included_runs": len(rows),
                "excluded_forced_or_stress_runs": excluded,
                "unreadable_reports": unreadable,
                "runs": rows,
                "aggregate_by_tool": aggregates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_markdown(rows, aggregates, excluded),
        encoding="utf-8",
    )

    print(f"candidate closed-loop runs: {len(candidates)}")
    print(f"included runs: {len(rows)}")
    print(f"excluded forced/stress runs: {excluded}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
