import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class RetinexformerLowLightEnhancer:
    def __init__(
        self,
        repo_dir,
        weights=None,
        config=None,
        device="cuda",
    ):
        self.repo_dir = Path(repo_dir).expanduser().resolve()

        self.weights = Path(
            weights
            or self.repo_dir
            / "pretrained_weights"
            / "LOL_v1.pth"
        ).expanduser().resolve()

        self.config = Path(
            config
            or self.repo_dir
            / "Options"
            / "RetinexFormer_LOL_v1.yml"
        ).expanduser().resolve()

        self.device = device

        if not self.repo_dir.is_dir():
            raise RuntimeError(
                f"Retinexformer仓库不存在：{self.repo_dir}"
            )

        if not self.weights.is_file():
            raise RuntimeError(
                f"Retinexformer权重不存在：{self.weights}"
            )

        if not self.config.is_file():
            raise RuntimeError(
                f"Retinexformer配置不存在：{self.config}"
            )

    def _load_model(self):
        repo_text = str(self.repo_dir)

        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)

        from basicsr.models.archs.RetinexFormer_arch import (
            RetinexFormer,
        )

        with self.config.open(
            "r",
            encoding="utf-8",
        ) as file:
            options = yaml.safe_load(file)

        network_options = dict(options["network_g"])
        network_options.pop("type", None)

        model = RetinexFormer(**network_options)

        checkpoint = torch.load(
            self.weights,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "params",
            checkpoint,
        )

        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model = model.to(self.device)
        model.eval()

        return model

    def _enhance_image(
        self,
        model,
        input_path,
        output_path,
    ):
        with Image.open(input_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(
                rgb,
                dtype=np.float32,
            ) / 255.0

        tensor = torch.from_numpy(array)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        height, width = tensor.shape[-2:]
        pad_height = (-height) % 4
        pad_width = (-width) % 4

        if pad_height or pad_width:
            tensor = F.pad(
                tensor,
                (0, pad_width, 0, pad_height),
                mode="reflect",
            )

        with torch.inference_mode():
            restored = model(tensor)

        restored = restored[
            :,
            :,
            :height,
            :width,
        ]

        restored = restored.clamp(0, 1)
        restored = restored.squeeze(0)
        restored = restored.permute(1, 2, 0)
        restored = restored.cpu().numpy()

        restored = np.rint(
            restored * 255.0
        ).astype(np.uint8)

        Image.fromarray(restored).save(output_path)

    def run_sequence(
        self,
        input_dir,
        output_dir,
    ):
        input_dir = Path(
            input_dir
        ).expanduser().resolve()

        output_dir = Path(
            output_dir
        ).expanduser().resolve()

        frame_paths = sorted(
            path
            for path in input_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )

        if not frame_paths:
            raise RuntimeError(
                f"输入目录没有图像：{input_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        started = time.perf_counter()

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        model = self._load_model()

        for input_path in frame_paths:
            output_path = output_dir / input_path.name

            self._enhance_image(
                model,
                input_path,
                output_path,
            )

        runtime_seconds = round(
            time.perf_counter() - started,
            3,
        )

        peak_gpu_memory_gb = None

        if self.device.startswith("cuda"):
            peak_gpu_memory_gb = round(
                torch.cuda.max_memory_allocated()
                / 1024**3,
                3,
            )

        output_frames = sum(
            1
            for path in output_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )

        del model

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "tool": "retinexformer_lowlight",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "input_frames": len(frame_paths),
            "output_frames": output_frames,
            "weights": str(self.weights),
            "config": str(self.config),
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }
