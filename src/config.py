"""
Configuration Management for Supernan AI Dubbing Pipeline

Centralized configuration for all pipeline components.
Supports both Colab and local environments.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch


class Config:
    """
    Central configuration class for the dubbing pipeline.
    
    Usage:
        from src.config import Config
        config = Config()
        print(config.WHISPER_MODEL)  # "large-v3"
    """
    
    # =====================================================================
    # PROJECT METADATA
    # =====================================================================
    VERSION = "1.0.0"
    AUTHOR = "Panchami K"
    DESCRIPTION = "Production-grade multilingual AI video dubbing"
    
    # =====================================================================
    # PATHS (Auto-detect Colab vs Local)
    # =====================================================================
    @property
    def BASE_DIR(self) -> Path:
        """Base directory of the project"""
        return Path(__file__).parent.parent.absolute()
    
    @property
    def SRC_DIR(self) -> Path:
        """Source code directory"""
        return self.BASE_DIR / "src"
    
    @property
    def DATA_DIR(self) -> Path:
        """Data directory (input/output/temp)"""
        return self.BASE_DIR / "data"
    
    @property
    def INPUT_DIR(self) -> Path:
        """Input videos directory"""
        return self.DATA_DIR / "input"
    
    @property
    def OUTPUT_DIR(self) -> Path:
        """Output videos directory"""
        return self.DATA_DIR / "output"
    
    @property
    def TEMP_DIR(self) -> Path:
        """Temporary files directory"""
        return self.DATA_DIR / "temp"
    
    @property
    def MODELS_DIR(self) -> Path:
        """Downloaded AI models directory"""
        return self.BASE_DIR / "models"
    
    @property
    def CHECKPOINTS_DIR(self) -> Path:
        """Pipeline checkpoints directory"""
        return self.BASE_DIR / "checkpoints"
    
    # =====================================================================
    # GPU / COMPUTE SETTINGS
    # =====================================================================
    @property
    def DEVICE(self) -> str:
        """Best available device (cuda > cpu)"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def GPU_NAME(self) -> Optional[str]:
        """Name of GPU if available"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return None
    
    FP16: bool = True  # Use half-precision for speed
    BATCH_SIZE: int = 1  # Conservative for Colab T4
    NUM_WORKERS: int = 2  # Parallel data loading
    
    # =====================================================================
    # VIDEO SETTINGS
    # =====================================================================
    VIDEO_FPS: int = 25  # Standard FPS
    VIDEO_CODEC: str = "libx264"  # H.264 codec
    AUDIO_CODEC: str = "aac"  # Audio codec
    AUDIO_SAMPLE_RATE: int = 22050  # XTTS optimal rate
    VIDEO_QUALITY: str = "high"  # Encoding quality
    
    # =====================================================================
    # SEGMENT SELECTION
    # =====================================================================
    DEFAULT_SEGMENT_DURATION: float = 15.0  # seconds
    MAX_SEGMENT_DURATION: float = 30.0  # seconds
    MIN_SEGMENT_DURATION: float = 10.0  # seconds
    
    # Content scoring weights (for AI segment selection)
    CONTENT_SCORE_MIN_LENGTH: float = 0.3  # Minimum text length
    CONTENT_SCORE_CENTER_BIAS: float = 0.2  # Prefer middle of video
    CONTENT_SCORE_KEYWORDS: List[str] = [  # Important content indicators
        "important", "main", "key", "remember", "note", "careful",
        "hygiene", "clean", "safety", "health", "child", "baby"
    ]
    
    # =====================================================================
    # WHISPER (Transcription)
    # =====================================================================
    WHISPER_MODEL: str = "large-v3"  # Model size (tiny/base/small/medium/large-v3)
    WHISPER_FALLBACK: str = "base"  # Fallback if OOM
    WHISPER_LANGUAGE: Optional[str] = None  # Auto-detect if None
    WHISPER_TASK: str = "transcribe"  # or "translate" (to English)
    
    # =====================================================================
    # TRANSLATION
    # =====================================================================
    TRANSLATION_MODEL: str = "AI4Bharat/IndicTrans2"  # Primary
    TRANSLATION_FALLBACK: str = "facebook/nllb-200-distilled-600M"  # Fallback
    
    # Supported language pairs (source -> target)
    SUPPORTED_LANGUAGES: Dict[str, Dict] = {
        "kn": {  # Kannada
            "name": "Kannada",
            "script": "Knda",
            "indic_code": "kan_Knda",
            "targets": ["hi", "en", "ta", "te"]
        },
        "hi": {  # Hindi
            "name": "Hindi",
            "script": "Deva",
            "indic_code": "hin_Deva",
            "targets": ["en"]
        },
        "en": {  # English
            "name": "English",
            "script": "Latn",
            "indic_code": "eng_Latn",
            "targets": ["hi", "kn", "ta", "te", "mr", "gu", "bn"]
        },
        "ta": {  # Tamil
            "name": "Tamil",
            "script": "Taml",
            "indic_code": "tam_Taml",
            "targets": ["hi", "en"]
        },
        "te": {  # Telugu
            "name": "Telugu",
            "script": "Telu",
            "indic_code": "tel_Telu",
            "targets": ["hi", "en"]
        },
        "mr": {  # Marathi
            "name": "Marathi",
            "script": "Deva",
            "indic_code": "mar_Deva",
            "targets": ["hi", "en"]
        },
        "gu": {  # Gujarati
            "name": "Gujarati",
            "script": "Gujr",
            "indic_code": "guj_Gujr",
            "targets": ["hi", "en"]
        }
    }
    
    DEFAULT_TARGET_LANGUAGE: str = "hi"  # Default: Hindi
    
    # =====================================================================
    # VOICE CLONING (XTTS)
    # =====================================================================
    XTTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    XTTS_SAMPLE_RATE: int = 22050
    XTTS_TEMPERATURE: float = 0.7  # Lower = more consistent
    XTTS_SPEED: float = 1.0  # Normal speed
    XTTS_REFERENCE_SECONDS: float = 6.0  # Seconds of reference audio needed
    
    # =====================================================================
    # LIP SYNC (Wav2Lip)
    # =====================================================================
    LIPSYNC_MODEL: str = "wav2lip"  # or "videoretalking"
    LIPSYNC_BATCH_SIZE: int = 1  # T4 constraint
    LIPSYNC_PAD_TOP: int = 0
    LIPSYNC_PAD_BOTTOM: int = 10
    LIPSYNC_PAD_LEFT: int = 0
    LIPSYNC_PAD_RIGHT: int = 0
    
    # =====================================================================
    # FACE RESTORATION (GFPGAN)
    # =====================================================================
    GFPGAN_MODEL: str = "GFPGANv1.4"
    GFPGAN_UPSCALE: int = 1  # No upscaling (save memory)
    GFPGAN_ONLY_CENTER_FACE: bool = True
    GFPGAN_BG_UPSAMPLER: Optional[str] = None  # No bg upsampling
    
    # =====================================================================
    # CHECKPOINTING
    # =====================================================================
    ENABLE_CHECKPOINTING: bool = True
    CHECKPOINT_INTERVAL: float = 30.0  # seconds of processed audio
    KEEP_ALL_CHECKPOINTS: bool = False  # Clean up to save space
    
    # =====================================================================
    # CACHE SETTINGS
    # =====================================================================
    ENABLE_CACHE: bool = True
    CACHE_DIR: Optional[Path] = None  # Auto-set in __init__
    
    # =====================================================================
    # LOGGING
    # =====================================================================
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: Optional[Path] = None  # Auto-set in __init__
    
    def __init__(self):
        """Initialize config and create necessary directories"""
        # Create directories if they don't exist
        for dir_path in [self.INPUT_DIR, self.OUTPUT_DIR, self.TEMP_DIR, 
                         self.MODELS_DIR, self.CHECKPOINTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set cache dir
        if self.ENABLE_CACHE and self.CACHE_DIR is None:
            self.CACHE_DIR = self.TEMP_DIR / "cache"
            self.CACHE_DIR.mkdir(exist_ok=True)
        
        # Set log file
        if self.LOG_FILE is None:
            self.LOG_FILE = self.TEMP_DIR / "pipeline.log"
    
    def get_cache_path(self, key: str) -> Path:
        """Get path for cached item"""
        if not self.ENABLE_CACHE or self.CACHE_DIR is None:
            return self.TEMP_DIR / key
        return self.CACHE_DIR / key
    
    def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.SUPPORTED_LANGUAGES
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full name of language"""
        if lang_code in self.SUPPORTED_LANGUAGES:
            return self.SUPPORTED_LANGUAGES[lang_code]["name"]
        return "Unknown"
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        return {
            "version": self.VERSION,
            "device": self.DEVICE,
            "gpu_name": self.GPU_NAME,
            "paths": {
                "base": str(self.BASE_DIR),
                "data": str(self.DATA_DIR),
                "output": str(self.OUTPUT_DIR),
                "models": str(self.MODELS_DIR)
            },
            "whisper_model": self.WHISPER_MODEL,
            "translation_model": self.TRANSLATION_MODEL,
            "xtts_model": self.XTTS_MODEL,
            "default_target": self.DEFAULT_TARGET_LANGUAGE
        }
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file"""
        if path is None:
            path = self.BASE_DIR / "config.json"
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Config saved to: {path}")
    
    def __repr__(self):
        return f"Config(device={self.DEVICE}, whisper={self.WHISPER_MODEL})"


# Global config instance (singleton pattern)
_config_instance = None

def get_config() -> Config:
    """Get global config instance (creates if needed)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


# For direct import
if __name__ == "__main__":
    config = Config()
    print("Configuration loaded successfully!")
    print("Device: " + str(config.DEVICE))
    print("GPU: " + str(config.GPU_NAME))
    print("Whisper Model: " + str(config.WHISPER_MODEL))
    print("Output Directory: " + str(config.OUTPUT_DIR))
    print("Supported languages: " + str(list(config.SUPPORTED_LANGUAGES.keys())))
    config.save()
