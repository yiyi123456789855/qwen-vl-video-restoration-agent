import argparse
import json
from collections import defaultdict
from pathlib import Path


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


def agreement_only_tool(report):
    raw_tool = report.get(
        "raw_model_selected_tool",
        report.get("raw_diagnosis", {}).get(
            "recommended_tool",
            "manual_review",
        ),
    )
    prior = report.get("objective_prior") or {}
    prior_tool = prior.get(
        "recommended_tool",
        "manual_review",
    )
    confidence = float(prior.get("confidence", 0.0))

    if (
        confidence >= 0.65
        and raw_tool == prior_tool
    ):
        return raw_tool

    return "manual_review"


def collect_cases(manifest, output_root, mode):
    cases = []
    missing = []

    for case in manifest["cases"]:
        report_path = (
            output_root
            / case["case_id"]
            / mode
            / "run_report.json"
        )
        if not report_path.is_file():
            missing.append(case["case_id"])
            continue

        report = load_json(report_path)
        raw_tool = report.get(
            "raw_model_selected_tool",
            report.get("raw_diagnosis", {}).get(
                "recommended_tool"
            ),
        )
        fused_v1_tool = report.get(
            "model_selected_tool",
            report.get("diagnosis", {}).get(
                "recommended_tool"
            ),
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "scene_id": case["scene_id"],
                "generation": case["generation"]["name"],
                "is_ood": case["is_ood"],
                "expected_tool": case["expected_tool"],
                "raw_vlm": raw_tool,
                "objective_override_v1": fused_v1_tool,
                "agreement_only_v2": agreement_only_tool(
                    report
                ),
            }
        )

    if missing:
        raise RuntimeError(
            f"缺少{len(missing)}个case结果："
            + ", ".join(missing)
        )
    return cases


def calculate_metrics(cases, policy):
    id_cases = [case for case in cases if not case["is_ood"]]
    ood_cases = [case for case in cases if case["is_ood"]]
    clean_cases = [
        case
        for case in cases
        if case["expected_tool"] == "none"
    ]
    accepted_id = [
        case
        for case in id_cases
        if case[policy] != "manual_review"
    ]

    return {
        "overall_tool_accuracy": safe_rate(
            sum(
                case[policy] == case["expected_tool"]
                for case in cases
            ),
            len(cases),
        ),
        "id_route_accuracy": safe_rate(
            sum(
                case[policy] == case["expected_tool"]
                for case in id_cases
            ),
            len(id_cases),
        ),
        "id_coverage": safe_rate(
            len(accepted_id),
            len(id_cases),
        ),
        "accepted_id_accuracy": safe_rate(
            sum(
                case[policy] == case["expected_tool"]
                for case in accepted_id
            ),
            len(accepted_id),
        ),
        "ood_rejection_rate": safe_rate(
            sum(
                case[policy] == "manual_review"
                for case in ood_cases
            ),
            len(ood_cases),
        ),
        "clean_passthrough_rate": safe_rate(
            sum(
                case[policy] == "none"
                for case in clean_cases
            ),
            len(clean_cases),
        ),
        "clean_false_activation_rate": safe_rate(
            sum(
                case[policy] in ACTIONABLE_TOOLS
                for case in clean_cases
            ),
            len(clean_cases),
        ),
        "manual_review_rate": safe_rate(
            sum(
                case[policy] == "manual_review"
                for case in cases
            ),
            len(cases),
        ),
    }


def failure_counts(cases, policy):
    failures = defaultdict(int)
    for case in cases:
        if case[policy] != case["expected_tool"]:
            failures[case["generation"]] += 1
    return dict(sorted(failures.items()))


def write_markdown(
    cases,
    policies,
    metrics,
    path,
):
    lines = [
        "# Routing Policy Comparison",
        "",
        "## Primary metrics",
        "",
        "| Metric | Raw VLM | Objective override v1 | Agreement only v2 |",
        "|---|---:|---:|---:|",
    ]
    labels = [
        ("overall_tool_accuracy", "Overall tool accuracy"),
        ("id_route_accuracy", "ID route accuracy"),
        ("id_coverage", "ID coverage"),
        (
            "accepted_id_accuracy",
            "Accepted ID route accuracy",
        ),
        ("ood_rejection_rate", "OOD rejection rate"),
        (
            "clean_passthrough_rate",
            "Clean passthrough rate",
        ),
        (
            "clean_false_activation_rate",
            "Clean false activation rate",
        ),
        ("manual_review_rate", "Manual review rate"),
    ]
    for key, label in labels:
        values = [
            percentage(metrics[policy][key])
            for policy in policies
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-case policy decisions",
            "",
            "| Case | Generation | OOD | Expected | Raw VLM | Override v1 | Agreement v2 |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case["case_id"],
                    case["generation"],
                    "yes" if case["is_ood"] else "no",
                    case["expected_tool"],
                    case["raw_vlm"],
                    case["objective_override_v1"],
                    case["agreement_only_v2"],
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
        default="results/routing_policy_comparison",
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
    cases = collect_cases(
        manifest,
        output_root,
        args.mode,
    )
    policies = [
        "raw_vlm",
        "objective_override_v1",
        "agreement_only_v2",
    ]
    metrics = {
        policy: calculate_metrics(cases, policy)
        for policy in policies
    }
    failures = {
        policy: failure_counts(cases, policy)
        for policy in policies
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "comparison.json"
    markdown_path = result_dir / "comparison.md"
    json_path.write_text(
        json.dumps(
            {
                "benchmark_id": manifest["benchmark_id"],
                "mode": args.mode,
                "case_count": len(cases),
                "metrics": metrics,
                "failure_counts": failures,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(
        cases,
        policies,
        metrics,
        markdown_path,
    )

    print(f"benchmark: {manifest['benchmark_id']}")
    print(f"cases: {len(cases)}")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
