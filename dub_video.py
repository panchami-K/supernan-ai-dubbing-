#!/usr/bin/env python3
"""CLI entry point for real visual lip-sync generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import module directly to avoid importing heavy optional deps from src/__init__.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lip_sync import LipSyncError, run_true_lipsync  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run true visual lip-sync with VideoReTalking")
    parser.add_argument("--face", required=True, help="Input face video path")
    parser.add_argument("--audio", required=True, help="Dubbed audio path")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument(
        "--vrt-dir",
        default="models/video-retalking",
        help="Path to local VideoReTalking checkout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out = run_true_lipsync(
            face_video=Path(args.face),
            dubbed_audio=Path(args.audio),
            output_video=Path(args.output),
            repo_dir=Path(args.vrt_dir),
        )
        print(f"✅ Real visual lip-sync completed: {out}")
        return 0
    except LipSyncError as exc:
        print(f"❌ Lip-sync failed: {exc}")
        print("No audio-only fallback was generated. Fix the model/runtime issue and rerun.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
