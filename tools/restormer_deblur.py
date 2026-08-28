import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def list_images(folder):
    folder = Path(folder)
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


class RestormerDeblurrer:
    def __init__(
        self,
        repo_dir,
        python_executable=None,
        tile=512,
        tile_overlap=64,
    ):
        self.repo_dir = Path(
            repo_dir
        ).expanduser().resolve()

        self.python_executable = (
            python_executable or sys.executable
        )
        self.tile = tile
        self.tile_overlap = tile_overlap

        self.demo_script = self.repo_dir / "demo.py"
        self.weights = (
            self.repo_dir
            / "Motion_Deblurring"
            / "pretrained_models"
            / "motion_deblurring.pth"
        )

        if not self.demo_script.is_file():
            raise FileNotFoundError(
                f"Restormer demo.py不存在："
                f"{self.demo_script}"
            )

        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Restormer去模糊权重不存在："
                f"{self.weights}"
            )

    def run_sequence(self, input_dir, output_dir):
        input_dir = Path(
            input_dir
        ).expanduser().resolve()

        output_dir = Path(
            output_dir
        ).expanduser().resolve()

        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"输入目录不存在：{input_dir}"
            )

        input_paths = list_images(input_dir)
        if not input_paths:
            raise RuntimeError(
                f"输入目录没有图像：{input_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        grayscale_stems = set()

        for path in input_paths:
            with Image.open(path) as image:
                if image.mode in {
                    "L", "I", "I;16"
                }:
                    grayscale_stems.add(path.stem)

        started = time.perf_counter()

        with tempfile.TemporaryDirectory(
            prefix="restormer_"
        ) as temp_dir:
            temp_root = Path(temp_dir)

            command = [
                str(self.python_executable),
                "demo.py",
                "--task",
                "Motion_Deblurring",
                "--input_dir",
                str(input_dir),
                "--result_dir",
                str(temp_root),
                "--tile",
                str(self.tile),
                "--tile_overlap",
                str(self.tile_overlap),
            ]

            completed = subprocess.run(
                command,
                cwd=self.repo_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    "Restormer运行失败：\n"
                    + completed.stdout
                )

            raw_output_dir = (
                temp_root / "Motion_Deblurring"
            )

            if not raw_output_dir.is_dir():
                raise RuntimeError(
                    "Restormer没有生成输出目录："
                    f"{raw_output_dir}"
                )

            for restored_path in list_images(
                raw_output_dir
            ):
                target_path = (
                    output_dir
                    / f"{restored_path.stem}.png"
                )

                if (
                    restored_path.stem
                    in grayscale_stems
                ):
                    with Image.open(
                        restored_path
                    ) as image:
                        image.convert("L").save(
                            target_path
                        )
                else:
                    shutil.copy2(
                        restored_path,
                        target_path,
                    )

        output_paths = list_images(output_dir)

        if len(output_paths) != len(input_paths):
            raise RuntimeError(
                "输入输出帧数不一致："
                f"输入{len(input_paths)}帧，"
                f"输出{len(output_paths)}帧"
            )

        return {
            "tool": "restormer_deblur",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "input_frames": len(input_paths),
            "output_frames": len(output_paths),
            "tile": self.tile,
            "tile_overlap": self.tile_overlap,
            "runtime_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }







