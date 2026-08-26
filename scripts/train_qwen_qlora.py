import argparse
import json
from pathlib import Path

import torch
from datasets import Image, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
    "--eval_steps",
    type=int,
    default=50,
    )
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=32,
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading datasets...", flush=True)
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_json,
            "validation": args.val_json,
        },
    )

    train_dataset = dataset["train"].cast_column(
        "image",
        Image(),
    )
    eval_dataset = dataset["validation"].cast_column(
        "image",
        Image(),
    )

    if args.smoke:
        train_count = min(
            args.max_train_samples,
            len(train_dataset),
        )
        eval_count = min(
            args.max_eval_samples,
            len(eval_dataset),
        )

        train_dataset = train_dataset.select(
            range(train_count)
        )
        eval_dataset = eval_dataset.select(
            range(eval_count)
        )

    print(
        f"train={len(train_dataset)}, "
        f"validation={len(eval_dataset)}",
        flush=True,
    )

    print("[2/6] Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("[3/6] Loading 4-bit base model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    max_steps = args.max_steps if args.smoke else -1
    eval_steps = (
    max(1, args.max_steps // 2)
    if args.smoke
    else args.eval_steps
    )

    training_config = SFTConfig(
        output_dir=str(output_dir),

        num_train_epochs=args.epochs,
        max_steps=max_steps,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        learning_rate=args.learning_rate,
        warmup_steps=1 if args.smoke else 10,
        lr_scheduler_type="cosine",

        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        logging_strategy="steps",
        logging_steps=1,

        eval_strategy="steps",
        eval_steps=eval_steps,

        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,

        max_grad_norm=1.0,
        max_length=None,
        packing=False,

        remove_unused_columns=False,
        dataset_num_proc=1,
        report_to="none",
        seed=args.seed,
    )

    print("[4/6] Creating trainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    print("[5/6] Trainable parameters:", flush=True)
    trainer.model.print_trainable_parameters()

    print("[6/6] Starting training...", flush=True)
    train_result = trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["validation_samples"] = len(eval_dataset)
    metrics["smoke"] = args.smoke
    metrics["lora_rank"] = args.lora_rank

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Adapter saved to: {final_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()