import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objective_prior import analyze_sequence


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = """
你是一个无参考图像质量诊断器。输入图像可能完全干净，也可能存在退化。

可选标签：
degradation只能选择：clean、noise、blur、jpeg、low_light、mixed、unknown
severity只能选择：none、mild、medium、severe
recommended_tool只能选择：
none、denoise、deblur、enhance_lowlight、manual_review
输入可能是单帧、多帧拼图，也可能是按时间顺序输入的原生视频帧序列。
如果是拼图，请综合所有子图，只判断跨帧持续出现的主要退化，
不要把拼图边界或用于填充的黑色区域当作图像退化。
如果是原生视频帧序列，请综合全部帧判断持续出现的主要退化。
判断原则：
1. 如果没有明确可见的退化，判断为clean、none、none。
2. 随机颗粒、孤立亮暗点或结构被随机噪声破坏，判断为noise，并选择denoise。
3. 边缘和纹理普遍不清晰、存在运动拖影，判断为blur，并选择deblur。
4. 出现明显块效应、振铃或压缩伪影，判断为jpeg，并选择manual_review。
5. 画面整体曝光明显不足、大面积区域过暗、暗部细节难以辨认，判断为low_light，并选择enhance_lowlight。
6. 仅有黑色背景、黑白色调、辐射图像低对比度，不足以判断为low_light。
7. 同时存在两种或以上明显退化时，判断为mixed，并选择manual_review。
8. 证据不足时判断为unknown，并选择manual_review。
9. clean的severity必须为none；其他退化的severity不能为none。
10. reason用不超过40个汉字说明可见依据。

只输出一个JSON对象，不要输出Markdown，不要复制固定答案。
JSON必须只包含以下四个键：
degradation、severity、recommended_tool、reason。
""".strip()


def parse_json(text: str) -> dict:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        return {
            "degradation": "unknown",
            "severity": "none",
            "recommended_tool": "none",
            "reason": f"无法解析模型输出：{text}",
        }

    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {
            "degradation": "unknown",
            "severity": "none",
            "recommended_tool": "none",
            "reason": f"JSON解析失败：{text}",
        }

    allowed_degradations = {
        "clean",
        "noise",
        "blur",
        "jpeg",
        "low_light",
        "mixed",
        "unknown",
    }
    allowed_severities = {"none", "mild", "medium", "severe"}
    allowed_tools = {
        "none",
        "denoise",
        "deblur",
        "enhance_lowlight",
        "manual_review",
    }

    if result.get("degradation") not in allowed_degradations:
        result["degradation"] = "unknown"

    if result.get("severity") not in allowed_severities:
        result["severity"] = "none"

    if result.get("recommended_tool") not in allowed_tools:
        result["recommended_tool"] = "none"

    result.setdefault("reason", "")
    return result


def compute_luminance_stats(image_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )

    return {
        "mean_luminance": round(
            float(luminance.mean()),
            4,
        ),
        "median_luminance": round(
            float(np.median(luminance)),
            4,
        ),
        "dark_pixel_ratio": round(
            float((luminance < 0.20).mean()),
            4,
        ),
        "p95_luminance": round(
            float(np.percentile(luminance, 95)),
            4,
        ),
    }


def fuse_diagnoses(raw_diagnosis, objective_prior):
    prior_confidence = float(
        objective_prior.get("confidence", 0.0)
    )
    prior_degradation = objective_prior.get(
        "degradation",
        "unknown",
    )
    prior_tool = objective_prior.get(
        "recommended_tool",
        "manual_review",
    )

    prior_is_actionable = (
        prior_degradation != "unknown"
        and prior_tool != "manual_review"
        and prior_confidence >= 0.65
    )

    if not prior_is_actionable:
        return (
            {
                "degradation": "unknown",
                "severity": "none",
                "recommended_tool": "manual_review",
                "confidence": round(prior_confidence, 4),
                "reason": "客观先验不确定，转人工复核",
            },
            "abstain_objective_uncertain",
        )

    agrees = (
        raw_diagnosis.get("degradation")
        == prior_degradation
        and raw_diagnosis.get("recommended_tool")
        == prior_tool
    )

    diagnosis = {
        "degradation": prior_degradation,
        "severity": objective_prior.get(
            "severity",
            "none",
        ),
        "recommended_tool": prior_tool,
        "confidence": round(prior_confidence, 4),
        "reason": objective_prior.get(
            "reason",
            "客观先验达到路由条件",
        ),
    }
    decision_source = (
        "vlm_objective_agreement"
        if agrees
        else "objective_prior_override"
    )
    return diagnosis, decision_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
    )
    parser.add_argument(
        "--video_frames",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--metric_frames",
        nargs="+",
        default=None,
        help="用于客观指标计算的原始采样帧",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )
    parser.add_argument(
        "--output",
        default="outputs/diagnosis.json",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="可选的LoRA Adapter目录",
    )
    args = parser.parse_args()

    if bool(args.image) == bool(args.video_frames):
        raise RuntimeError(
            "--image和--video_frames必须且只能提供一个"
        )

    video_paths = None

    if args.video_frames:
        video_paths = [
            Path(path).expanduser().resolve()
            for path in args.video_frames
        ]

        missing_paths = [
            path
            for path in video_paths
            if not path.is_file()
        ]

        if missing_paths:
            raise FileNotFoundError(
                f"视频帧不存在：{missing_paths}"
            )

        image_path = video_paths[
            len(video_paths) // 2
        ]
        input_type = "video"
    else:
        image_path = (
            Path(args.image)
            .expanduser()
            .resolve()
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"图像不存在：{image_path}"
            )

        input_type = "image"

    if args.metric_frames:
        metric_paths = [
            Path(path).expanduser().resolve()
            for path in args.metric_frames
        ]
    elif video_paths is not None:
        metric_paths = video_paths
    else:
        metric_paths = [image_path]

    missing_metric_paths = [
        path
        for path in metric_paths
        if not path.is_file()
    ]
    if missing_metric_paths:
        raise FileNotFoundError(
            f"客观指标帧不存在：{missing_metric_paths}"
        )

    objective_report = analyze_sequence(
        metric_paths
    )
    objective_prior = objective_report[
        "objective_prior"
    ]

    luminance_stats = compute_luminance_stats(
        image_path
    )

    mean_luminance = luminance_stats[
        "mean_luminance"
    ]
    dark_pixel_ratio = luminance_stats[
        "dark_pixel_ratio"
    ]

    if (
        mean_luminance < 0.15
        and dark_pixel_ratio > 0.90
    ):
        brightness_hint = (
            "strong_low_light_candidate"
        )
    elif (
        mean_luminance < 0.25
        and dark_pixel_ratio > 0.60
    ):
        brightness_hint = (
            "possible_low_light_candidate"
        )
    else:
        brightness_hint = "normal_or_uncertain"

    diagnostic_prompt = f"""
{PROMPT}

程序计算的辅助亮度统计如下，数值范围为0到1：
mean_luminance={luminance_stats["mean_luminance"]}
median_luminance={luminance_stats["median_luminance"]}
dark_pixel_ratio={luminance_stats["dark_pixel_ratio"]}
p95_luminance={luminance_stats["p95_luminance"]}
brightness_hint={brightness_hint}

如果brightness_hint是strong_low_light_candidate，
且画面是自然场景，应优先判断为low_light并选择
enhance_lowlight。若图像是有意使用黑色背景的辐射、
医学或灰度图像，则亮度统计不能单独作为低照度证据。
""".strip()

    print(
        "[Brightness]",
        json.dumps(
            luminance_stats,
            ensure_ascii=False,
        ),
        brightness_hint,
        flush=True,
    )
    print(
        "[Objective prior]",
        json.dumps(
            objective_prior,
            ensure_ascii=False,
        ),
        flush=True,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading processor...", flush=True)
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

    print("[2/4] Loading Qwen2.5-VL in NF4...", flush=True)
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
        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"Adapter目录不存在：{adapter_path}"
            )

        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(
                f"Adapter配置不存在：{adapter_config}"
            )

        print(f"[Adapter] Loading: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    model.eval()

    if video_paths is None:
        visual_input = {
            "type": "image",
            "path": str(image_path),
        }
    else:
        visual_input = {
            "type": "video",
            "video": [
                path.as_uri()
                for path in video_paths
            ],
            "fps": 1.0,
            "min_pixels": 128 * 28 * 28,
            "max_pixels": 256 * 28 * 28,
        }

    conversation = [
        {
            "role": "user",
            "content": [
                visual_input,
                {
                    "type": "text",
                    "text": diagnostic_prompt,
                },
            ],
        }
    ]

    print("[3/4] Processing visual input...", flush=True)

    if video_paths is None:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        prompt_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        (
            image_inputs,
            video_inputs,
            video_kwargs,
        ) = process_vision_info(
            conversation,
            return_video_kwargs=True,
        )

        fps_value = video_kwargs.get("fps")

        if isinstance(
            fps_value,
            (list, tuple),
        ):
            video_kwargs["fps"] = fps_value[0]

        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    print("[4/4] Generating diagnosis...", flush=True)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    raw_diagnosis = parse_json(raw_text)
    diagnosis, decision_source = fuse_diagnoses(
        raw_diagnosis,
        objective_prior,
    )
    report = {
        "image": str(image_path),
        "input_type": input_type,
        "video_frames": (
            None
            if video_paths is None
            else [
                str(path)
                for path in video_paths
            ]
        ),
        "luminance_stats": luminance_stats,
        "brightness_hint": brightness_hint,
        "metric_frames": [
            str(path)
            for path in metric_paths
        ],
        "objective_features": objective_report[
            "features"
        ],
        "objective_prior": objective_prior,
        "model": args.model,
        "adapter": (
            None if adapter_path is None else str(adapter_path)
        ),
        "raw_diagnosis": raw_diagnosis,
        "diagnosis": diagnosis,
        "decision_source": decision_source,
        "raw_output": raw_text,
        "inference_seconds": round(elapsed, 3),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRaw output:")
    print(raw_text)
    print("\nRaw parsed diagnosis:")
    print(
        json.dumps(
            raw_diagnosis,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nFused diagnosis:")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    print(f"Decision source: {decision_source}")
    print(f"\nSaved to: {output_path}")
    print(f"Inference time: {elapsed:.3f} seconds")
    print(
        "Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB"
    )


if __name__ == "__main__":
    main()
