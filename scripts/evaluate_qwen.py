import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen_diagnoser import PROMPT, parse_json


LABELS = ["clean", "noise"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strict_json_valid(text):
    try:
        value = json.loads(text.strip())
        return isinstance(value, dict)
    except json.JSONDecodeError:
        return False


def cleaned_json_valid(text):
    cleaned = (
        text.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end <= start:
        return False

    try:
        value = json.loads(cleaned[start : end + 1])
        return isinstance(value, dict)
    except json.JSONDecodeError:
        return False


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def macro_f1(targets, predictions):
    scores = {}

    for label in LABELS:
        tp = sum(
            target == label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        fp = sum(
            target != label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        fn = sum(
            target == label and prediction != label
            for target, prediction in zip(targets, predictions)
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(
            2 * precision * recall,
            precision + recall,
        )

        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro = sum(
        item["f1"] for item in scores.values()
    ) / len(scores)

    return macro, scores


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0

    index = round((len(values) - 1) * fraction)
    return values[index]


def calculate_metrics(predictions):
    total = len(predictions)

    targets = [row["target_degradation"] for row in predictions]
    predicted = [
        row["predicted_degradation"] for row in predictions
    ]

    degradation_correct = sum(
        target == prediction
        for target, prediction in zip(targets, predicted)
    )

    tool_correct = sum(
        row["target_tool"] == row["predicted_tool"]
        for row in predictions
    )

    severity_correct = sum(
        row["target_severity"] == row["predicted_severity"]
        for row in predictions
    )

    noise_rows = [
        row for row in predictions
        if row["target_degradation"] == "noise"
    ]
    clean_rows = [
        row for row in predictions
        if row["target_degradation"] == "clean"
    ]

    noise_severity_correct = sum(
        row["target_severity"] == row["predicted_severity"]
        for row in noise_rows
    )

    clean_false_triggers = sum(
        row["predicted_tool"] != "none"
        for row in clean_rows
    )

    noise_tool_hits = sum(
        row["predicted_tool"] == "denoise"
        for row in noise_rows
    )

    strict_valid = sum(
        row["strict_json_valid"] for row in predictions
    )
    cleaned_valid = sum(
        row["cleaned_json_valid"] for row in predictions
    )

    latencies = [
        row["latency_seconds"] for row in predictions
    ]

    macro, per_class = macro_f1(targets, predicted)

    return {
        "samples": total,
        "degradation_accuracy": safe_div(
            degradation_correct,
            total,
        ),
        "degradation_macro_f1": macro,
        "per_class": per_class,
        "tool_accuracy": safe_div(tool_correct, total),
        "severity_accuracy_all": safe_div(
            severity_correct,
            total,
        ),
        "severity_accuracy_noise_only": safe_div(
            noise_severity_correct,
            len(noise_rows),
        ),
        "clean_false_trigger_rate": safe_div(
            clean_false_triggers,
            len(clean_rows),
        ),
        "noise_tool_recall": safe_div(
            noise_tool_hits,
            len(noise_rows),
        ),
        "strict_json_valid_rate": safe_div(
            strict_valid,
            total,
        ),
        "cleaned_json_valid_rate": safe_div(
            cleaned_valid,
            total,
        ),
        "prediction_counts": dict(Counter(predicted)),
        "severity_prediction_counts": dict(
            Counter(
                row["predicted_severity"]
                for row in predictions
            )
        ),
        "latency_mean_seconds": (
            statistics.mean(latencies) if latencies else 0.0
        ),
        "latency_median_seconds": (
            statistics.median(latencies) if latencies else 0.0
        ),
        "latency_p95_seconds": percentile(latencies, 0.95),
    }


def main():
    args = parse_args()

    rows = read_jsonl(args.test_json)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluation samples: {len(rows)}", flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    adapter_path = None
    if args.adapter:
        adapter_path = Path(args.adapter).expanduser().resolve()
        print(f"Loading adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    model.eval()
    torch.cuda.reset_peak_memory_stats()

    predictions = []
    total_start = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image"]).resolve()

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        start = time.perf_counter()

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        latency = time.perf_counter() - start

        generated_ids = output_ids[
            :,
            inputs.input_ids.shape[1] :,
        ]
        raw_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        diagnosis = parse_json(raw_text)

        predictions.append(
            {
                "image": str(image_path),
                "sequence": row.get("sequence"),
                "frame": row.get("frame"),
                "pair_psnr": row.get("pair_psnr"),
                "target_degradation": row["label"],
                "target_severity": row["severity"],
                "target_tool": row["recommended_tool"],
                "predicted_degradation": diagnosis["degradation"],
                "predicted_severity": diagnosis["severity"],
                "predicted_tool": diagnosis["recommended_tool"],
                "reason": diagnosis["reason"],
                "raw_output": raw_text,
                "strict_json_valid": strict_json_valid(raw_text),
                "cleaned_json_valid": cleaned_json_valid(raw_text),
                "latency_seconds": round(latency, 4),
            }
        )

        if index % 10 == 0 or index == len(rows):
            print(
                f"[{index}/{len(rows)}] "
                f"last={latency:.3f}s",
                flush=True,
            )

    total_runtime = time.perf_counter() - total_start
    metrics = calculate_metrics(predictions)
    metrics.update(
        {
            "model": args.model,
            "adapter": (
                None
                if adapter_path is None
                else str(adapter_path)
            ),
            "total_runtime_seconds": round(total_runtime, 3),
            "peak_gpu_memory_gb": round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            ),
        }
    )

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()