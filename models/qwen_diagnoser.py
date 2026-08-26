import argparse
import json
import time
from pathlib import Path

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
degradation只能选择：clean、noise、blur、jpeg、low_light、unknown
severity只能选择：none、mild、medium、severe
recommended_tool只能选择：
none、denoise、deblur、dejpeg、enhance_low_light

判断原则：
1. 如果没有明确可见的退化，必须判断为clean、none、none。
2. 只有看到明显随机颗粒、孤立亮暗点或结构被随机噪声破坏时，才判断为noise。
3. 灰度图、黑白色调、辐射图像本身的纹理和较低对比度，不等于存在噪声。
4. 不要因为任务涉及图像复原，就默认图像一定需要处理。
5. 选择最主要的一种退化；证据不足时选择unknown和none。
6. reason用不超过40个汉字说明可见依据。

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
        "clean", "noise", "blur", "jpeg", "low_light", "unknown"
    }
    allowed_severities = {"none", "mild", "medium", "severe"}
    allowed_tools = {
        "none", "denoise", "deblur", "dejpeg", "enhance_low_light"
    }

    if result.get("degradation") not in allowed_degradations:
        result["degradation"] = "unknown"

    if result.get("severity") not in allowed_severities:
        result["severity"] = "none"

    if result.get("recommended_tool") not in allowed_tools:
        result["recommended_tool"] = "none"

    result.setdefault("reason", "")
    return result


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
                    "text": PROMPT,
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
