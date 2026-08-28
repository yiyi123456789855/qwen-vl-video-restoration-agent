import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODE_ORDER = {
    "single": 0,
    "contact_sheet": 1,
    "native_video": 2,
}

EXPECTED_TOOLS = {
    "clean": "none",
    "noise": "denoise",
    "noisy": "denoise",
    "blur": "deblur",
    "low_light": "enhance_lowlight",
    "lowlight": "enhance_lowlight",
}

CSV_FIELDS = [
    "case",
    "mode",
    "degradation",
    "severity",
    "expected_tool",
    "selected_tool",
    "route_correct",
    "diagnosis_seconds",
    "peak_gpu_memory_gb",
    "tool_runtime_seconds",
    "total_runtime_seconds",
    "run_report",
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_tool_report(run_report):
    for key in ("denoise", "deblur", "lowlight"):
        value = run_report.get(key)
        if isinstance(value, dict):
            return value
    return {}


def collect_rows(outputs_dir):
    rows = []

    for run_path in outputs_dir.rglob("run_report.json"):
        run_report = load_json(run_path)
        mode = run_report.get("diagnosis_mode", "single")

        if mode not in MODE_ORDER:
            continue

        if run_report.get("force_tool") is not None:
            continue

        diagnosis_path = run_path.parent / "diagnosis.json"
        diagnosis_report = (
            load_json(diagnosis_path)
            if diagnosis_path.is_file()
            else {}
        )

        diagnosis = run_report.get("diagnosis", {})
        case_name = Path(
            run_report.get("input_dir", run_path.parent.name)
        ).name

        expected_tool = EXPECTED_TOOLS.get(case_name)
        selected_tool = run_report.get(
            "model_selected_tool",
            run_report.get("selected_tool"),
        )
        route_correct = (
            None
            if expected_tool is None
            else selected_tool == expected_tool
        )

        tool_report = first_tool_report(run_report)

        rows.append(
            {
                "case": case_name,
                "mode": mode,
                "degradation": diagnosis.get("degradation"),
                "severity": diagnosis.get("severity"),
                "expected_tool": expected_tool,
                "selected_tool": selected_tool,
                "route_correct": route_correct,
                "diagnosis_seconds": diagnosis_report.get(
                    "inference_seconds"
                ),
                "peak_gpu_memory_gb": diagnosis_report.get(
                    "peak_gpu_memory_gb"
                ),
                "tool_runtime_seconds": tool_report.get(
                    "runtime_seconds"
                ),
                "total_runtime_seconds": run_report.get(
                    "total_runtime_seconds"
                ),
                "run_report": str(run_path.resolve()),
            }
        )

    rows.sort(
        key=lambda row: (
            row["case"],
            MODE_ORDER[row["mode"]],
            row["run_report"],
        )
    )
    return rows


def select_comparable_rows(rows):
    latest_by_case_and_mode = {}

    for row in rows:
        key = (row["case"], row["mode"])
        modified_time = Path(
            row["run_report"]
        ).stat().st_mtime

        previous = latest_by_case_and_mode.get(key)

        if (
            previous is None
            or modified_time > previous[0]
        ):
            latest_by_case_and_mode[key] = (
                modified_time,
                row,
            )

    modes_by_case = defaultdict(set)

    for case_name, mode in latest_by_case_and_mode:
        modes_by_case[case_name].add(mode)

    required_modes = set(MODE_ORDER)
    comparable_cases = {
        case_name
        for case_name, modes in modes_by_case.items()
        if modes == required_modes
    }

    selected = [
        row
        for (case_name, _), (_, row)
        in latest_by_case_and_mode.items()
        if case_name in comparable_cases
    ]

    selected.sort(
        key=lambda row: (
            row["case"],
            MODE_ORDER[row["mode"]],
        )
    )
    return selected, sorted(comparable_cases)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate_by_mode(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["mode"]].append(row)

    summaries = []

    for mode in sorted(grouped, key=MODE_ORDER.get):
        mode_rows = grouped[mode]
        scored = [
            row
            for row in mode_rows
            if row["route_correct"] is not None
        ]
        correct = sum(
            row["route_correct"] is True
            for row in scored
        )

        summaries.append(
            {
                "mode": mode,
                "runs": len(mode_rows),
                "scored_runs": len(scored),
                "correct_runs": correct,
                "route_accuracy": (
                    correct / len(scored)
                    if scored
                    else None
                ),
                "mean_diagnosis_seconds": mean(
                    [
                        row["diagnosis_seconds"]
                        for row in mode_rows
                    ]
                ),
                "mean_peak_gpu_memory_gb": mean(
                    [
                        row["peak_gpu_memory_gb"]
                        for row in mode_rows
                    ]
                ),
                "mean_total_runtime_seconds": mean(
                    [
                        row["total_runtime_seconds"]
                        for row in mode_rows
                    ]
                ),
            }
        )

    return summaries


def display(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, summaries, path):
    lines = [
        "# Diagnosis Mode Comparison",
        "",
        "## Per-run results",
        "",
        "| Case | Mode | Degradation | Expected | Selected | Correct | Diagnosis s | Peak GB | Total s |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    display(row["case"]),
                    display(row["mode"]),
                    display(row["degradation"]),
                    display(row["expected_tool"]),
                    display(row["selected_tool"]),
                    display(row["route_correct"]),
                    display(row["diagnosis_seconds"]),
                    display(row["peak_gpu_memory_gb"]),
                    display(row["total_runtime_seconds"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate by mode",
            "",
            "| Mode | Runs | Accuracy | Mean diagnosis s | Mean peak GB | Mean total s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for summary in summaries:
        accuracy = summary["route_accuracy"]
        accuracy_text = (
            "-"
            if accuracy is None
            else f"{accuracy * 100:.1f}%"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["mode"],
                    str(summary["runs"]),
                    accuracy_text,
                    display(summary["mean_diagnosis_seconds"]),
                    display(summary["mean_peak_gpu_memory_gb"]),
                    display(summary["mean_total_runtime_seconds"]),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs_dir",
        default="outputs",
    )
    parser.add_argument(
        "--result_dir",
        default="results/diagnosis_mode_comparison",
    )
    parser.add_argument(
        "--include_incomplete",
        action="store_true",
        help=(
            "保留未覆盖全部三种模式的case；"
            "默认仅比较三种模式齐全的case"
        ),
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    result_dir = Path(args.result_dir).expanduser().resolve()

    if not outputs_dir.is_dir():
        raise FileNotFoundError(
            f"输出目录不存在：{outputs_dir}"
        )

    candidate_rows = collect_rows(outputs_dir)

    if not candidate_rows:
        raise RuntimeError(
            f"没有找到可用的run_report.json：{outputs_dir}"
        )

    comparable_cases = []

    if args.include_incomplete:
        rows = candidate_rows
    else:
        rows, comparable_cases = select_comparable_rows(
            candidate_rows
        )

        if not rows:
            raise RuntimeError(
                "没有case同时包含single、contact_sheet和"
                "native_video三种自动运行结果"
            )

    summaries = aggregate_by_mode(rows)
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / "comparison.csv"
    json_path = result_dir / "comparison.json"
    markdown_path = result_dir / "comparison.md"

    write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "runs": rows,
                "aggregate_by_mode": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(rows, summaries, markdown_path)

    print(f"candidate runs: {len(candidate_rows)}")
    print(f"selected runs: {len(rows)}")

    if comparable_cases:
        print(
            "comparable cases: "
            + ", ".join(comparable_cases)
        )

    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
