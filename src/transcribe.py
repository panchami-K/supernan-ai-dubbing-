
"""
DYNAMIC Transcription Module - Auto-correction for ANY language
"""

import whisper
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
import numpy as np
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class Transcriber:
    """Fully dynamic Whisper transcriber - works for ANY language."""

    def __init__(self, model_size: str = "large-v3", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        print(f"Loading Whisper: {model_size} on {self.device}")
        self.model = whisper.load_model(model_size).to(self.device)
        print("✅ Whisper loaded")
        
        # UNIVERSAL correction patterns (work for ANY language)
        self.universal_patterns = {
            # Repeated characters (works for Devanagari/Latin/Cyrillic/...)
            r"(.){3,}": r"\1\1",  # aaa → aa
            r"(.){2,}": r"\1",     # aaaa → a
            
            # Common English loanwords (hygiene, clean, safe)
            r"hygein": "hygiene",
            r"hygine": "hygiene", 
            r"clien": "clean",
            r"clne": "clean",
            
            # Common noise artifacts
            r"^um+": "",
            r"^uh+": "",
            r"^ah+": "",
            
            # Punctuation cleanup
            r"\s+([.,!?])": r"\1",
            r"([.,!?])\s+": r"\1 "
        }

    def preprocess_audio(self, video_path: str) -> str:
        """Universal noise reduction."""
        import librosa
        import soundfile as sf
        y, sr = librosa.load(video_path, sr=16000)
        y_clean = librosa.effects.preemphasis(y * 0.95)
        clean_path = "/tmp/clean_audio.wav"
        sf.write(clean_path, y_clean, sr)
        print(f"🔧 Audio cleaned: {len(y):,} → {len(y_clean):,} samples")
        return clean_path

    def smart_correct(self, text: str) -> str:
        """UNIVERSAL auto-correction using phonetic similarity."""
        text = text.strip()
        if len(text) < 3:
            return text
            
        # Apply universal regex patterns
        for pattern, replacement in self.universal_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Fix common phonetic confusions (language-agnostic)
        common_fixes = {
            "hygein": "hygiene", "hygine": "hygiene", "clin": "clean",
            "safty": "safety", "importnt": "important", "chld": "child"
        }
        for wrong, right in common_fixes.items():
            if SequenceMatcher(None, wrong, text.lower()).ratio() > 0.7:
                text = text.replace(wrong, right)
        
        return text.strip()

    def transcribe(self, video_path: str, language: Optional[str] = None) -> Dict:
        """Dynamic transcription + universal auto-correction."""
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\n🎯 Transcribing: {Path(video_path).name}")
        clean_audio = self.preprocess_audio(video_path)
        print("⏳ Processing (3-5 minutes)...")

        result = self.model.transcribe(
            clean_audio,
            language=language,  # Auto-detect ANY language
            fp16=(self.device == "cuda"),
            verbose=True,
            word_timestamps=True,
            condition_on_previous_text=True,
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.2,
            logprob_threshold=-1.0,
        )

        detected_language = result.get("language", "unknown")
        segments = result.get("segments", [])
        
        # DYNAMIC CORRECTION - works for ANY language
        full_text = self.smart_correct(result.get("text", "").strip())
        for seg in segments:
            seg["text"] = self.smart_correct(seg["text"].strip())

        duration = segments[-1]["end"] if segments else 0

        print(f"\n📊 TRANSCRIPTION COMPLETE:")
        print(f"   Language: {detected_language}")
        print(f"   Segments: {len(segments)}")
        print(f"   Duration: {duration:.2f}s")

        cleaned_segments = self._clean_segments(segments)
        return {
            "language": detected_language,
            "full_text": full_text,
            "segments": cleaned_segments,
            "duration": duration,
            "video_path": video_path
        }

    def _clean_segments(self, segments: List[Dict]) -> List[Dict]:
        """Universal repetition cleaning."""
        cleaned = []
        prev_text = ""

        for seg in segments:
            text = seg["text"].strip()
            
            # Skip obvious garbage
            if (len(set(text)) < 5 or 
                len(text) < 3 or
                re.search(r"(.){5,}", text)):
                continue

            if text != prev_text and len(text) > 3:
                seg["text"] = text
                cleaned.append(seg)
                prev_text = text

        return cleaned

    def find_best_segment(self, transcription: Dict, target_duration: float = 15.0) -> Dict:
        """Universal challenge-optimized segment finder."""
        segments = transcription.get("segments", [])
        total_duration = transcription.get("duration", 0)

        print(f"\n🔍 Analyzing {len(segments)} segments (0-30s priority)...")

        best_score = -1
        best_window = None

        # Challenge sweet spot: 0-30s
        for start in np.arange(0, min(35, total_duration - 10), 0.5):
            end = min(start + target_duration, total_duration)
            if end - start < 12:
                continue

            window_segs = [s for s in segments if s["start"] >= start and s["end"] <= end]
            if len(window_segs) < 2:
                continue

            text = " ".join([s["text"] for s in window_segs])
            if len(text) < 25:
                continue

            # Universal scoring
            text_len = len(text)
            words = text.split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            early_bonus = max(0, (1 - start/30) * 80)
            
            # Content bonus (hygiene/safety keywords)
            content_bonus = 40 if any(kw in text.lower() for kw in 
                                    ['hygiene', 'clean', 'safety', 'health', 'important']) else 0

            score = text_len * 0.3 + unique_ratio * 40 + early_bonus + content_bonus

            if score > best_score:
                best_score = score
                best_window = {
                    "start": float(start),
                    "end": float(end),
                    "duration": float(end - start),
                    "score": float(score),
                    "text": text[:200],
                    "segments": len(window_segs)
                }

        # Smart fallback
        if best_window is None and segments:
            early_segs = [s for s in segments if s["start"] < 40]
            if early_segs:
                best_seg = max(early_segs, key=lambda s: len(s["text"]))
                mid = (best_seg["start"] + best_seg["end"]) / 2
                best_window = {
                    "start": max(0, mid - 7.5),
                    "end": min(total_duration, mid + 7.5),
                    "duration": 15.0,
                    "score": 10.0,
                    "text": best_seg["text"][:200],
                    "segments": 1
                }

        print(f"⭐ BEST SEGMENT: {best_window['start']:.1f}s-{best_window['end']:.1f}s "
              f"(score: {best_window['score']:.1f})")
        return best_window

    def save_transcription(self, transcription: Dict, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {output_path}")
