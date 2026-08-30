import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


# Gradio performs a localhost startup check. Make sure a machine-level proxy
# does not intercept that request.
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
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

STATUS_LABELS = {
    "accept": "✅ 自动接受",
    "manual_review": "🟠 转人工复核",
    "retry": "🔁 自动重试",
    "stop": "🛑 安全停止",
    "diagnosis_only": "🔎 仅诊断",
    None: "—",
}

TOOL_LABELS = {
    "denoise": "五帧视频去噪",
    "deblur": "Restormer 去模糊",
    "enhance_lowlight": "Retinexformer 低照增强",
    "none": "原图直通",
    "manual_review": "人工复核",
    None: "—",
}


def natural_key(path: Path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_key,
    )


def uploaded_path(item):
    if isinstance(item, (str, Path)):
        return Path(item).expanduser().resolve()
    name = getattr(item, "name", None)
    if name:
        return Path(name).expanduser().resolve()
    raise gr.Error(f"无法识别上传文件：{item!r}")


def resolve_sources(
    files,
    server_sequence_dir,
    allowed_input_root,
    max_input_frames,
):
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

        sources = list_images(source_dir)
        if not sources:
            raise gr.Error(f"服务器目录中没有图像：{source_dir}")
    else:
        if not files:
            raise gr.Error(
                "请上传连续图像帧，或填写服务器序列目录。"
            )
        sources = sorted(
            [uploaded_path(item) for item in files],
            key=natural_key,
        )

    if len(sources) > max_input_frames:
        raise gr.Error(
            f"输入共 {len(sources)} 帧，超过网页上限 "
            f"{max_input_frames} 帧。"
        )
    return sources


def stage_sources(sources, input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    staged = []

    for index, source in enumerate(sources):
        if not source.is_file():
            raise gr.Error(f"输入文件不存在：{source}")
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise gr.Error(f"不支持的图像格式：{source.name}")

        target = input_dir / f"{index:06d}.png"
        try:
            with Image.open(source) as image:
                # V2 must preserve colour information for low-light, JPEG and
                # unknown colour-cast diagnosis. The V1 UI converted to L mode.
                image.convert("RGB").save(target)
        except Exception as exc:
            raise gr.Error(f"无法读取图像 {source.name}：{exc}") from exc
        staged.append(target)

    return staged


def optional_resolved_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def add_optional_argument(command, flag, value):
    resolved = optional_resolved_path(value)
    if resolved is not None:
        command.extend([flag, resolved])


def build_pipeline_command(
    args,
    input_dir,
    output_dir,
    diagnosis_mode,
    diagnosis_frames,
):
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
        "--denoise_test_script",
        str(Path(args.denoise_test_script).expanduser().resolve()),
        "--denoise_weights",
        str(Path(args.denoise_weights).expanduser().resolve()),
        "--device",
        args.device,
        "--cuda_visible_devices",
        args.cuda_visible_devices,
        "--diagnosis_mode",
        diagnosis_mode,
        "--diagnosis_frames",
        str(int(diagnosis_frames)),
        "--fusion_policy",
        args.fusion_policy,
        "--quality_max_attempts",
        str(args.quality_max_attempts),
    ]

    add_optional_argument(command, "--qwen_adapter", args.qwen_adapter)
    add_optional_argument(command, "--restormer_repo", args.restormer_repo)
    add_optional_argument(
        command,
        "--restormer_python",
        args.restormer_python,
    )
    add_optional_argument(
        command,
        "--retinexformer_repo",
        args.retinexformer_repo,
    )
    add_optional_argument(
        command,
        "--retinexformer_weights",
        args.retinexformer_weights,
    )
    add_optional_argument(
        command,
        "--retinexformer_config",
        args.retinexformer_config,
    )
    return command


def make_environment(args):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.allow_model_download:
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
    else:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def quality_check_rows(quality):
    if not isinstance(quality, dict):
        return []

    rows = []
    for check in quality.get("checks", []):
        if not isinstance(check, dict):
            continue
        rows.append(
            [
                check.get("name", ""),
                "通过" if check.get("passed") else "未通过",
                check.get("actual"),
                check.get("operator", ""),
                check.get("threshold"),
                check.get("description", ""),
            ]
        )
    return rows


def create_download_bundle(run_dir, output_dir, published_frames):
    export_dir = run_dir / "download"
    restored_export = export_dir / "restored"
    restored_export.mkdir(parents=True, exist_ok=True)

    for frame in published_frames:
        shutil.copy2(frame, restored_export / frame.name)

    for name in (
        "diagnosis.json",
        "run_report.json",
    ):
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, export_dir / name)

    log_path = run_dir / "pipeline.log"
    if log_path.is_file():
        shutil.copy2(log_path, export_dir / log_path.name)

    archive_base = run_dir / f"v2_result_{run_dir.name}"
    return Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=export_dir,
        )
    )


def build_summary(report, run_id, input_count):
    diagnosis = report.get("diagnosis") or {}
    quality = report.get("quality") or {}
    selected_tool = report.get("selected_tool")
    final_status = report.get("closed_loop_status")

    return {
        "run_id": run_id,
        "diagnosis_mode": report.get("diagnosis_mode"),
        "input_frames": input_count,
        "degradation": diagnosis.get("degradation"),
        "severity": diagnosis.get("severity"),
        "raw_model_tool": report.get("raw_model_selected_tool"),
        "objective_prior_tool": (
            (report.get("objective_prior") or {}).get("recommended_tool")
        ),
        "selected_tool": selected_tool,
        "selected_tool_label": TOOL_LABELS.get(
            selected_tool,
            selected_tool,
        ),
        "decision_source": report.get("decision_source"),
        "routing_confidence": report.get("routing_confidence"),
        "final_status": final_status,
        "final_status_label": STATUS_LABELS.get(
            final_status,
            final_status,
        ),
        "quality_score": quality.get("quality_score"),
        "attempt_count": len(report.get("restoration_attempts") or []),
        "published_result": report.get("published_result"),
        "total_runtime_seconds": report.get("total_runtime_seconds"),
    }


def status_markdown(report):
    diagnosis = report.get("diagnosis") or {}
    tool = report.get("selected_tool")
    status = report.get("closed_loop_status")
    confidence = report.get("routing_confidence")
    confidence_text = "—" if confidence is None else f"{confidence:.3f}"

    return (
        f"## {STATUS_LABELS.get(status, status or '运行完成')}\n\n"
        f"- **退化类型：** `{diagnosis.get('degradation', 'unknown')}`\n"
        f"- **严重程度：** `{diagnosis.get('severity', 'none')}`\n"
        f"- **最终工具：** `{tool}`（{TOOL_LABELS.get(tool, tool)}）\n"
        f"- **决策来源：** `{report.get('decision_source', 'unknown')}`\n"
        f"- **路由置信度：** `{confidence_text}`\n"
        f"- **总耗时：** `{report.get('total_runtime_seconds', '—')} s`"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_script", required=True)
    parser.add_argument(
        "--qwen_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--qwen_adapter",
        default=None,
        help="可选；不传时使用 Zero-shot 基座模型",
    )
    parser.add_argument("--denoise_test_script", required=True)
    parser.add_argument("--denoise_weights", required=True)
    parser.add_argument(
        "--restormer_repo",
        default="third_party/Restormer",
    )
    parser.add_argument(
        "--restormer_python",
        default=sys.executable,
    )
    parser.add_argument(
        "--retinexformer_repo",
        default="third_party/Retinexformer",
    )
    parser.add_argument("--retinexformer_weights", default=None)
    parser.add_argument("--retinexformer_config", default=None)
    parser.add_argument("--runs_dir", default="outputs/gradio_v2_runs")
    parser.add_argument(
        "--allowed_input_root",
        default=None,
        help="可选；限制网页可读取的服务器目录根路径",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument(
        "--fusion_policy",
        default="agreement_only",
    )
    parser.add_argument("--quality_max_attempts", type=int, default=2)
    parser.add_argument("--max_input_frames", type=int, default=300)
    parser.add_argument(
        "--pipeline_timeout_seconds",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="允许从 Hugging Face 下载缺失模型；默认强制离线",
    )
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建临时公开 Gradio 链接；不要用于敏感数据",
    )
    return parser.parse_args()


def build_demo(args):
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    allowed_input_root = (
        Path(args.allowed_input_root).expanduser().resolve()
        if args.allowed_input_root
        else None
    )

    def run_pipeline(
        uploaded_files,
        server_sequence_dir,
        diagnosis_mode,
        diagnosis_frames,
    ):
        run_id = uuid.uuid4().hex[:12]
        run_dir = runs_dir / run_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "pipeline"

        sources = resolve_sources(
            uploaded_files,
            server_sequence_dir,
            allowed_input_root,
            args.max_input_frames,
        )
        staged = stage_sources(sources, input_dir)

        command = build_pipeline_command(
            args,
            input_dir,
            output_dir,
            diagnosis_mode,
            diagnosis_frames,
        )
        environment = make_environment(args)

        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.pipeline_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            partial_log = exc.stdout or ""
            if isinstance(partial_log, bytes):
                partial_log = partial_log.decode("utf-8", errors="replace")
            log_path = run_dir / "pipeline.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(partial_log, encoding="utf-8")
            raise gr.Error(
                f"Pipeline 超过 {args.pipeline_timeout_seconds} 秒，已停止等待。"
            ) from exc

        log_path = run_dir / "pipeline.log"
        log_path.write_text(completed.stdout, encoding="utf-8")

        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            raise gr.Error(f"Pipeline 运行失败：\n{tail}")

        report_path = output_dir / "run_report.json"
        report = load_json(report_path)
        if not isinstance(report, dict):
            raise gr.Error(f"Pipeline 没有生成有效报告：{report_path}")

        restored = list_images(output_dir / "restored")
        published_frames = restored if restored else staged
        used_original_fallback = not bool(restored)

        if restored and len(restored) != len(staged):
            raise gr.Error(
                f"输出帧数异常：输入 {len(staged)} 帧，"
                f"输出 {len(restored)} 帧"
            )

        middle = len(staged) // 2
        output_middle = min(middle, len(published_frames) - 1)

        gallery = []
        for index, path in enumerate(published_frames):
            label = f"输出 {index + 1:03d}"
            if used_original_fallback:
                label = f"原图回退 {index + 1:03d}"
            gallery.append((str(path), label))

        archive_path = create_download_bundle(
            run_dir,
            output_dir,
            published_frames,
        )

        quality = report.get("quality") or {}
        attempts = report.get("restoration_attempts") or []
        summary = build_summary(report, run_id, len(staged))

        return (
            status_markdown(report),
            str(staged[middle]),
            str(published_frames[output_middle]),
            summary,
            report.get("raw_diagnosis") or {},
            report.get("objective_prior") or {},
            report.get("diagnosis") or {},
            quality_check_rows(quality),
            attempts,
            gallery,
            str(report_path),
            str(archive_path),
            completed.stdout,
        )

    css = """
    .hero {text-align: center; margin-bottom: 0.8rem;}
    .hero h1 {font-size: 2rem; margin-bottom: 0.25rem;}
    .hero p {color: #64748b;}
    .result-card {border-radius: 12px;}
    """

    with gr.Blocks(
        title="Memory-Aware Video Restoration Agent",
        css=css,
    ) as demo:
        gr.Markdown(
            "<div class='hero'>"
            "<h1>Memory-Aware Multimodal Video Restoration Agent</h1>"
            "<p>Qwen2.5-VL · Objective Prior · Safe Routing · "
            "Multi-tool Restoration · Closed-loop Quality Control</p>"
            "</div>"
        )

        gr.Markdown(
            "上传连续视频帧。系统将展示 VLM 原始判断、客观先验、"
            "融合路由、工具执行过程与质量门控结果。遇到未知退化、"
            "证据冲突或质量风险时允许拒答并转人工复核。"
        )

        with gr.Row():
            with gr.Column(scale=2):
                uploads = gr.File(
                    label="上传连续图像帧",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                server_sequence = gr.Textbox(
                    label="或填写服务器序列目录",
                    placeholder="/path/to/ordered/frames",
                )
            with gr.Column(scale=1):
                diagnosis_mode = gr.Dropdown(
                    choices=[
                        ("单帧（最终冻结配置）", "single"),
                        ("多帧拼图", "contact_sheet"),
                        ("原生视频帧输入", "native_video"),
                    ],
                    value="single",
                    label="诊断模式",
                )
                diagnosis_frames = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=5,
                    step=1,
                    label="均匀采样帧数",
                )
                run_button = gr.Button(
                    "开始诊断与复原",
                    variant="primary",
                )

        status_output = gr.Markdown(
            "## 等待输入",
            elem_classes=["result-card"],
        )

        with gr.Row():
            input_preview = gr.Image(
                label="代表输入帧",
                type="filepath",
            )
            output_preview = gr.Image(
                label="最终发布帧",
                type="filepath",
            )

        with gr.Tabs():
            with gr.Tab("决策总览"):
                summary_output = gr.JSON(label="运行与最终决策")

            with gr.Tab("诊断证据"):
                with gr.Row():
                    raw_output = gr.JSON(label="Raw VLM diagnosis")
                    prior_output = gr.JSON(label="Objective prior")
                    fused_output = gr.JSON(label="Fused diagnosis")

            with gr.Tab("质量闭环"):
                quality_table = gr.Dataframe(
                    headers=[
                        "检查项",
                        "结果",
                        "实际值",
                        "关系",
                        "阈值",
                        "说明",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "number",
                        "str",
                        "number",
                        "str",
                    ],
                    interactive=False,
                    label="最终质量门控",
                )
                attempts_output = gr.JSON(label="全部复原尝试")

            with gr.Tab("全部输出"):
                gallery_output = gr.Gallery(
                    label="最终发布序列",
                    columns=5,
                    object_fit="contain",
                )

            with gr.Tab("报告与日志"):
                with gr.Row():
                    report_download = gr.File(
                        label="下载 run_report.json"
                    )
                    bundle_download = gr.File(
                        label="下载完整结果 ZIP"
                    )
                log_output = gr.Textbox(
                    label="Pipeline 日志",
                    lines=16,
                    max_lines=32,
                )

        gr.Markdown(
            "**说明：** `manual_review` 是系统的安全输出，不代表程序失败。"
            "公开临时链接前请确保输入不含敏感数据。"
        )

        run_button.click(
            fn=run_pipeline,
            inputs=[
                uploads,
                server_sequence,
                diagnosis_mode,
                diagnosis_frames,
            ],
            outputs=[
                status_output,
                input_preview,
                output_preview,
                summary_output,
                raw_output,
                prior_output,
                fused_output,
                quality_table,
                attempts_output,
                gallery_output,
                report_download,
                bundle_download,
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
        share=args.share,
    )


if __name__ == "__main__":
    main()
