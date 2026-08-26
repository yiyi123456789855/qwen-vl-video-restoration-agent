import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_test_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "video_test1210_runtime",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载测试脚本：{script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tensor(path: Path, channels: int) -> torch.Tensor:
    with Image.open(path) as image:
        if channels == 1:
            array = np.asarray(
                image.convert("L"),
                dtype=np.uint8,
            )[:, :, None]
        elif channels == 3:
            array = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"不支持的通道数：{channels}")

    return (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
    )


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]

    array = (
        tensor.mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if array.shape[0] == 1:
        Image.fromarray(array[0], mode="L").save(path)
    elif array.shape[0] == 3:
        Image.fromarray(
            np.transpose(array, (1, 2, 0)),
            mode="RGB",
        ).save(path)
    else:
        raise ValueError(f"无法保存的张量形状：{array.shape}")


class VideoDenoiser:
    def __init__(
        self,
        test_script: str,
        weights: str,
        device: str = "cuda",
        frames: int = 5,
        channels: int = 1,
        base: int = 64,
        tile: int = 512,
        overlap: int = 128,
    ):
        self.script_path = Path(test_script).expanduser().resolve()
        self.weights_path = Path(weights).expanduser().resolve()

        if not self.script_path.is_file():
            raise FileNotFoundError(
                f"测试脚本不存在：{self.script_path}"
            )
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"去噪权重不存在：{self.weights_path}"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了CUDA，但PyTorch未检测到GPU")
        if overlap < 0 or overlap >= tile:
            raise ValueError("overlap必须满足0 <= overlap < tile")

        self.device = device
        self.tile = tile
        self.overlap = overlap
        self.runtime = load_test_module(self.script_path)

        checkpoint = torch.load(
            self.weights_path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.frames = int(checkpoint.get("frames", frames))
            self.channels = int(
                checkpoint.get(
                    "channels",
                    1 if checkpoint.get("gray", channels == 1) else 3,
                )
            )
            self.base = int(checkpoint.get("base", base))
            state_dict = checkpoint["state_dict"]
        else:
            self.frames = frames
            self.channels = channels
            self.base = base
            state_dict = checkpoint

        if self.frames < 1 or self.frames % 2 == 0:
            raise ValueError(
                f"frames必须是正奇数，实际为{self.frames}"
            )

        self.radius = self.frames // 2
        self.model = self.runtime.VideoStackUNet(
            self.channels,
            self.frames,
            self.base,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def run_sequence(self, input_dir: str, output_dir: str) -> dict:
        source_dir = Path(input_dir).expanduser().resolve()
        target_dir = Path(output_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"输入序列不存在：{source_dir}")

        frame_paths = self.runtime.list_images(source_dir)
        if not frame_paths:
            raise RuntimeError(f"序列中没有图像：{source_dir}")

        target_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        with torch.inference_mode():
            for center, center_path in enumerate(frame_paths):
                indices = [
                    min(max(center + offset, 0), len(frame_paths) - 1)
                    for offset in range(-self.radius, self.radius + 1)
                ]
                clip = torch.stack(
                    [
                        load_tensor(frame_paths[index], self.channels)
                        for index in indices
                    ],
                    dim=0,
                )
                prediction = self.runtime.infer_tiled(
                    self.model,
                    clip,
                    self.device,
                    tile=self.tile,
                    overlap=self.overlap,
                )
                save_tensor(prediction, target_dir / center_path.name)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        return {
            "input_dir": str(source_dir),
            "output_dir": str(target_dir),
            "weights": str(self.weights_path),
            "frames": self.frames,
            "channels": self.channels,
            "base": self.base,
            "tile": self.tile,
            "overlap": self.overlap,
            "output_frames": len(frame_paths),
            "runtime_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_script", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    denoiser = VideoDenoiser(
        test_script=args.test_script,
        weights=args.weights,
        device=args.device,
        frames=args.frames,
        channels=args.channels,
        base=args.base,
        tile=args.tile,
        overlap=args.overlap,
    )
    report = denoiser.run_sequence(
        args.input_dir,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
