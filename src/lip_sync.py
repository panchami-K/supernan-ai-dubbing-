"""Utilities to run **real** visual lip-sync with VideoReTalking/Wav2Lip.

This module intentionally avoids fake mouth drawing fallbacks. If the model
cannot run, it raises a clear error so callers know visual lip-sync did not
happen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Checkpoint:
    """A required checkpoint file for VideoReTalking."""

    filename: str
    url: str
    min_size_bytes: int = 1_000_000


REQUIRED_CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        "30_net_gen.pth",
        "https://huggingface.co/spaces/Plachta/VideoReTalking/resolve/main/checkpoints/30_net_gen.pth",
    ),
    Checkpoint(
        "wav2lip.pth",
        "https://huggingface.co/spaces/Plachta/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
    ),
    Checkpoint(
        "detection_Resnet50_Final.pth",
        "https://huggingface.co/spaces/Plachta/VideoReTalking/resolve/main/checkpoints/detection_Resnet50_Final.pth",
    ),
    Checkpoint(
        "shape_predictor_68_face_landmarks.dat",
        "https://huggingface.co/spaces/Plachta/VideoReTalking/resolve/main/checkpoints/shape_predictor_68_face_landmarks.dat",
    ),
)


class LipSyncError(RuntimeError):
    """Raised when real visual lip sync cannot be completed."""


def _run(cmd: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise LipSyncError(f"Missing required binary: {name}")


def _download(url: str, target: Path) -> None:
    _require_binary("wget")
    result = _run(["wget", "-q", "--show-progress", "-O", str(target), url])
    if result.returncode != 0:
        raise LipSyncError(f"Failed downloading {url}\n{result.stderr.strip()}")


def ensure_videoretalking_repo(repo_dir: Path) -> None:
    """Clone VideoReTalking if missing."""
    if repo_dir.exists() and (repo_dir / "inference.py").exists():
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _require_binary("git")
    result = _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/OpenTalker/video-retalking.git",
            str(repo_dir),
        ]
    )
    if result.returncode != 0:
        raise LipSyncError(f"Unable to clone VideoReTalking\n{result.stderr.strip()}")


def ensure_checkpoints(repo_dir: Path, checkpoints: Iterable[Checkpoint] = REQUIRED_CHECKPOINTS) -> None:
    """Download required checkpoints if absent/corrupt."""
    ckpt_dir = repo_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in checkpoints:
        target = ckpt_dir / ckpt.filename
        if target.exists() and target.stat().st_size >= ckpt.min_size_bytes:
            continue
        _download(ckpt.url, target)
        if not target.exists() or target.stat().st_size < ckpt.min_size_bytes:
            raise LipSyncError(f"Checkpoint invalid after download: {target}")


def run_true_lipsync(face_video: Path, dubbed_audio: Path, output_video: Path, repo_dir: Path) -> Path:
    """Run VideoReTalking inference for true visual lip-sync.

    Raises LipSyncError if the model fails. No silent audio-only fallback.
    """
    _require_binary("ffmpeg")

    if not face_video.exists():
        raise LipSyncError(f"Missing input video: {face_video}")
    if not dubbed_audio.exists():
        raise LipSyncError(f"Missing input audio: {dubbed_audio}")

    ensure_videoretalking_repo(repo_dir)
    ensure_checkpoints(repo_dir)

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_dir / "inference.py"),
        "--face",
        str(face_video),
        "--audio",
        str(dubbed_audio),
        "--outfile",
        str(output_video),
        "--tmp_dir",
        str(repo_dir / "temp"),
    ]

    result = _run(cmd, cwd=repo_dir)
    if result.returncode != 0:
        raise LipSyncError(
            "VideoReTalking failed.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    if not output_video.exists() or output_video.stat().st_size < 1_000_000:
        raise LipSyncError(f"Lip-sync output missing or too small: {output_video}")

    return output_video
