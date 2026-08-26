import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# Gradio启动时会通过HTTP访问自己的startup-events接口。服务器若设置了
# HTTP(S)_PROXY而没有排除localhost，该自检请求可能被代理转发并返回502。
for proxy_bypass_name in ("NO_PROXY", "no_proxy"):
    proxy_bypass = os.environ.get(proxy_bypass_name, "")
    entries = [item.strip() for item in proxy_bypass.split(",") if item.strip()]
    for localhost_entry in ("127.0.0.1", "localhost"):
        if localhost_entry not in entries:
            entries.append(localhost_entry)
    os.environ[proxy_bypass_name] = ",".join(entries)

import gradio as gr
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


def natural_key(path: Path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def resolve_sources(files, server_sequence_dir, allowed_input_root):
    server_sequence_dir = (server_sequence_dir or "").strip()
    if server_sequence_dir:
        source_dir = Path(server_sequence_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise gr.Error(f"服务器序列目录不存在：{source_dir}")
        if allowed_input_root is not None:
            try:
                source_dir.relative_to(allowed_input_root)
            except ValueError as exc:
                raise gr.Error(
                    f"服务器目录必须位于：{allowed_input_root}"
                ) from exc
        sources = sorted(
            [
                path for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ],
            key=natural_key,
        )
        if not sources:
            raise gr.Error(f"服务器目录中没有图像：{source_dir}")
        return sources

    if not files:
        raise gr.Error(
            "请上传图像，或填写服务器序列目录；建议使用连续的5帧或更多帧。"
        )
    return [Path(item).expanduser().resolve() for item in files]


def stage_sources(sources, input_dir: Path):

    input_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise gr.Error(f"上传文件不存在：{source}")
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise gr.Error(f"不支持的图像格式：{source.name}")

        target = input_dir / f"{index:06d}.png"
        try:
            with Image.open(source) as image:
                image.convert("L").save(target)
        except Exception as exc:
            raise gr.Error(f"无法读取图像 {source.name}：{exc}") from exc
        staged.append(target)
    return staged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_script", required=True)
    parser.add_argument(
        "--qwen_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--qwen_adapter", required=True)
    parser.add_argument("--denoise_test_script", required=True)
    parser.add_argument("--denoise_weights", required=True)
    parser.add_argument("--runs_dir", default="outputs/gradio_runs")
    parser.add_argument(
        "--allowed_input_root",
        default=None,
        help="可选；限制界面可读取的服务器目录根路径",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


def build_demo(args):
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    allowed_input_root = (
        Path(args.allowed_input_root).expanduser().resolve()
        if args.allowed_input_root
        else None
    )

    def run_pipeline(uploaded_files, server_sequence_dir):
        run_id = uuid.uuid4().hex[:12]
        run_dir = runs_dir / run_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "pipeline"
        sources = resolve_sources(
            uploaded_files,
            server_sequence_dir,
            allowed_input_root,
        )
        staged = stage_sources(sources, input_dir)

        command = [
            sys.executable,
            "-u",
            "-m",
            "app.pipeline",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--qwen_script",
            str(Path(args.qwen_script).expanduser().resolve()),
            "--qwen_model",
            args.qwen_model,
            "--qwen_adapter",
            str(Path(args.qwen_adapter).expanduser().resolve()),
            "--denoise_test_script",
            str(Path(args.denoise_test_script).expanduser().resolve()),
            "--denoise_weights",
            str(Path(args.denoise_weights).expanduser().resolve()),
            "--device",
            args.device,
            "--cuda_visible_devices",
            args.cuda_visible_devices,
        ]
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path = run_dir / "pipeline.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-20:])
            raise gr.Error(f"Pipeline运行失败：\n{tail}")

        report_path = output_dir / "run_report.json"
        if not report_path.is_file():
            raise gr.Error(f"Pipeline没有生成报告：{report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))

        restored_dir = output_dir / "restored"
        restored = sorted(
            [
                path for path in restored_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ],
            key=natural_key,
        )
        if len(restored) != len(staged):
            raise gr.Error(
                f"输出帧数异常：输入{len(staged)}帧，输出{len(restored)}帧"
            )

        middle = len(staged) // 2
        archive_base = run_dir / f"restored_{run_id}"
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=restored_dir,
            )
        )

        diagnosis = report.get("diagnosis", {})
        summary = {
            "run_id": run_id,
            "selected_tool": report.get("selected_tool"),
            "action": report.get("action"),
            "total_runtime_seconds": report.get("total_runtime_seconds"),
            "input_frames": len(staged),
            "diagnosis": diagnosis,
        }
        gallery = [(str(path), path.name) for path in restored]
        return (
            str(staged[middle]),
            str(restored[middle]),
            summary,
            gallery,
            str(archive_path),
            completed.stdout,
        )

    with gr.Blocks(title="Qwen2.5-VL低质视频诊断与自动复原") as demo:
        gr.Markdown(
            "# Qwen2.5-VL低质视频诊断与自动复原\n"
            "上传连续灰度帧。系统先诊断代表帧，再自动选择去噪或原图直通。"
        )
        uploads = gr.File(
            label="上传连续图像帧（建议至少5帧）",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
        server_sequence = gr.Textbox(
            label="或填写服务器序列目录",
            placeholder=(
                "/path/to/video_denoise/"
                "radiation_crvd_scenes1_11_video_noise_compact_v5/"
                "test/noisy/scene10__ISO12800__srcavi_pair"
            ),
        )
        run_button = gr.Button("开始诊断与复原", variant="primary")

        with gr.Row():
            input_preview = gr.Image(label="代表输入帧", type="filepath")
            output_preview = gr.Image(label="代表输出帧", type="filepath")

        diagnosis_output = gr.JSON(label="诊断与工具调用结果")
        gallery_output = gr.Gallery(
            label="全部输出帧",
            columns=5,
            object_fit="contain",
        )
        download_output = gr.File(label="下载全部输出帧（ZIP）")
        log_output = gr.Textbox(
            label="运行日志",
            lines=12,
            max_lines=24,
        )

        run_button.click(
            fn=run_pipeline,
            inputs=[uploads, server_sequence],
            outputs=[
                input_preview,
                output_preview,
                diagnosis_output,
                gallery_output,
                download_output,
                log_output,
            ],
        )

    return demo, runs_dir


def main():
    args = parse_args()
    demo, runs_dir = build_demo(args)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        allowed_paths=[str(runs_dir)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
