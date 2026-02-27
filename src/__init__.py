"""
Supernan AI Dubbing Pipeline
Production-grade multilingual video dubbing
"""

__version__ = "1.0.0"
__author__ = "Panchami K"

from .config import Config, get_config
from .video_utils import VideoProcessor, AudioProcessor, get_video_info, extract_segment
from .transcribe import Transcriber, quick_transcribe

__all__ = [
    "Config",
    "get_config",
    "VideoProcessor",
    "AudioProcessor",
    "get_video_info",
    "extract_segment",
    "Transcriber",
    "quick_transcribe"
]
