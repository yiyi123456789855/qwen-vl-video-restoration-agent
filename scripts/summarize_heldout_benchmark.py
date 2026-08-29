import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CSV_FIELDS = [
    "case_id",
    "scene_id",
    "generation",
    "is_ood",
    "expected_degradation",
    "expected_tool",
    "raw_degradation",
    "raw_tool",
    "raw_correct",
    "fused_degradation",
    "fused_tool",
    "fused_correct",
    "decision_source",
    "routing_confidence",
    "diagnosis_seconds",
    "peak_gpu_memory_gb",
    "total_runtime_seconds",
    "run_report",
]


ACTIONABLE_TOOLS = {
    "denoise",
    "deblur",
    "enhance_lowlight",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_rate(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def percentage(value):
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def display(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect_rows(
    project_root,
    manifest,
    output_root,
    mode,
):
    rows = []
    missing = []

    for case in manifest["cases"]:
        run_path = (
            output_root
            / case["case_id"]
            / mode
            / "run_report.json"
        )
        if not run_path.is_file():
            missing.append(case["case_id"])
            continue

        report = load_json(run_path)
        diagnosis_path = run_path.parent / "diagnosis.json"
        diagnosis_report = (
            load_json(diagnosis_path)
            if diagnosis_path.is_file()
            else {}
        )
        raw = report.get(
            "raw_diagnosis",
            report.get("diagnosis", {}),
        )
        fused = report.get("diagnosis", {})
        raw_tool = report.get(
            "raw_model_selected_tool",
            raw.get("recommended_tool"),
        )
        fused_tool = report.get(
            "model_selected_tool",
            fused.get("recommended_tool"),
        )
        expected_tool = case["expected_tool"]

        rows.append(
            {
                "case_id": case["case_id"],
                "scene_id": case["scene_id"],
                "generation": case["generation"]["name"],
                "is_ood": case["is_ood"],
                "expected_degradation": case[
                    "expected_degradation"
                ],
                "expected_tool": expected_tool,
                "raw_degradation": raw.get("degradation"),
                "raw_tool": raw_tool,
                "raw_correct": raw_tool == expected_tool,
                "fused_degradation": fused.get("degradation"),
                "fused_tool": fused_tool,
                "fused_correct": fused_tool == expected_tool,
                "decision_source": report.get(
                    "decision_source"
                ),
                "routing_confidence": report.get(
                    "routing_confidence"
                ),
                "diagnosis_seconds": diagnosis_report.get(
                    "inference_seconds"
                ),
                "peak_gpu_memory_gb": diagnosis_report.get(
                    "peak_gpu_memory_gb"
                ),
                "total_runtime_seconds": report.get(
                    "total_runtime_seconds"
                ),
                "run_report": str(
                    run_path.relative_to(project_root)
                ),
            }
        )

    return rows, missing


def calculate_metrics(rows, tool_field):
    correct_field = (
        "raw_correct"
        if tool_field == "raw_tool"
        else "fused_correct"
    )
    id_rows = [row for row in rows if not row["is_ood"]]
    ood_rows = [row for row in rows if row["is_ood"]]
    clean_rows = [
        row
        for row in rows
        if row["expected_degradation"] == "clean"
    ]

    return {
        "overall_tool_accuracy": safe_rate(
            sum(row[correct_field] for row in rows),
            len(rows),
        ),
        "id_route_accuracy": safe_rate(
            sum(row[correct_field] for row in id_rows),
            len(id_rows),
        ),
        "ood_rejection_rate": safe_rate(
            sum(
                row[tool_field] == "manual_review"
                for row in ood_rows
            ),
            len(ood_rows),
        ),
        "clean_passthrough_rate": safe_rate(
            sum(
                row[tool_field] == "none"
                for row in clean_rows
            ),
            len(clean_rows),
        ),
        "clean_false_activation_rate": safe_rate(
            sum(
                row[tool_field] in ACTIONABLE_TOOLS
                for row in clean_rows
            ),
            len(clean_rows),
        ),
        "manual_review_rate": safe_rate(
            sum(
                row[tool_field] == "manual_review"
                for row in rows
            ),
            len(rows),
        ),
    }


def metrics_by_generation(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["generation"]].append(row)

    summaries = []
    for generation in sorted(grouped):
        group = grouped[generation]
        summaries.append(
            {
                "generation": generation,
                "runs": len(group),
                "raw_accuracy": safe_rate(
                    sum(row["raw_correct"] for row in group),
                    len(group),
                ),
                "fused_accuracy": safe_rate(
                    sum(row["fused_correct"] for row in group),
                    len(group),
                ),
            }
        )
    return summaries


def write_csv(rows, path):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows,
    raw_metrics,
    fused_metrics,
    generation_summaries,
    source_counts,
    path,
):
    lines = [
        "# Held-out Benchmark Report",
        "",
        "## Primary metrics",
        "",
        "| Metric | Raw VLM | Fused routing |",
        "|---|---:|---:|",
    ]
    metric_labels = [
        ("overall_tool_accuracy", "Overall tool accuracy"),
        ("id_route_accuracy", "ID route accuracy"),
        ("ood_rejection_rate", "OOD rejection rate"),
        ("clean_passthrough_rate", "Clean passthrough rate"),
        (
            "clean_false_activation_rate",
            "Clean false activation rate",
        ),
        ("manual_review_rate", "Manual review rate"),
    ]
    for key, label in metric_labels:
        lines.append(
            f"| {label} | "
            f"{percentage(raw_metrics[key])} | "
            f"{percentage(fused_metrics[key])} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy by generation",
            "",
            "| Generation | Runs | Raw accuracy | Fused accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in generation_summaries:
        lines.append(
            f"| {item['generation']} | {item['runs']} | "
            f"{percentage(item['raw_accuracy'])} | "
            f"{percentage(item['fused_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Decision sources",
            "",
            "| Source | Runs |",
            "|---|---:|",
        ]
    )
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")

    lines.extend(
        [
            "",
            "## Per-case results",
            "",
            "| Case | Generation | OOD | Expected | Raw | Fused | Raw correct | Fused correct | Source | Confidence |",
            "|---|---|---:|---|---|---|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["generation"],
                    display(row["is_ood"]),
                    row["expected_tool"],
                    display(row["raw_tool"]),
                    display(row["fused_tool"]),
                    display(row["raw_correct"]),
                    display(row["fused_correct"]),
                    display(row["decision_source"]),
                    display(row["routing_confidence"]),
                ]
            )
            + " |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
        "--result_dir",
        default="results/heldout_v1",
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=[
            "single",
            "contact_sheet",
            "native_video",
        ],
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    manifest_path = (
        project_root / args.manifest
    ).resolve()
    output_root = (
        project_root / args.output_root
    ).resolve()
    result_dir = (
        project_root / args.result_dir
    ).resolve()

    manifest = load_json(manifest_path)
    rows, missing = collect_rows(
        project_root,
        manifest,
        output_root,
        args.mode,
    )

    if missing:
        raise RuntimeError(
            f"缺少{len(missing)}个case结果："
            + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("没有可汇总的benchmark结果")

    raw_metrics = calculate_metrics(rows, "raw_tool")
    fused_metrics = calculate_metrics(rows, "fused_tool")
    generation_summaries = metrics_by_generation(rows)
    source_counts = Counter(
        row["decision_source"] or "missing"
        for row in rows
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "comparison.csv"
    json_path = result_dir / "comparison.json"
    markdown_path = result_dir / "comparison.md"

    write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "benchmark_id": manifest["benchmark_id"],
                "mode": args.mode,
                "runs": len(rows),
                "raw_metrics": raw_metrics,
                "fused_metrics": fused_metrics,
                "accuracy_by_generation": generation_summaries,
                "decision_sources": dict(source_counts),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(
        rows,
        raw_metrics,
        fused_metrics,
        generation_summaries,
        source_counts,
        markdown_path,
    )

    print(f"benchmark: {manifest['benchmark_id']}")
    print(f"mode: {args.mode}")
    print(f"runs: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
