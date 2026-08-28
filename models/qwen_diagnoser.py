import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = """
你是一个无参考图像质量诊断器。输入图像可能完全干净，也可能存在退化。

可选标签：
degradation只能选择：clean、noise、blur、jpeg、low_light、mixed、unknown
severity只能选择：none、mild、medium、severe
recommended_tool只能选择：
none、denoise、deblur、enhance_lowlight、manual_review
输入可能是单帧，也可能是按时间顺序排列的多帧拼图。
如果是拼图，请综合所有子图，只判断跨帧持续出现的主要退化，
不要把拼图边界或用于填充的黑色区域当作图像退化。
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
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

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图像不存在：{image_path}")
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
                    "text": diagnostic_prompt,
                },
            ],
        }
    ]

    print("[3/4] Processing image...", flush=True)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
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

    diagnosis = parse_json(raw_text)
    report = {
        "image": str(image_path),
        "luminance_stats": luminance_stats,
        "brightness_hint": brightness_hint,
        "model": args.model,
        "adapter": (
            None if adapter_path is None else str(adapter_path)
        ),
        "diagnosis": diagnosis,
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
    print("\nParsed result:")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")
    print(f"Inference time: {elapsed:.3f} seconds")
    print(
        "Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB"
    )


if __name__ == "__main__":
    main()
