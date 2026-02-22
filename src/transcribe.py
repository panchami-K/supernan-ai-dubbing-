"""
Transcription Module - Speech-to-Text with Whisper

Auto-detects language and transcribes video with timestamped segments.
Includes AI-powered segment selection for finding the best 15-30 seconds.
"""

import whisper
import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


class Transcriber:
    """
    Whisper-based transcriber with auto-language detection.
    
    Usage:
        from src.transcribe import Transcriber
        t = Transcriber()
        result = t.transcribe("video.mp4")
        best_segment = t.find_best_segment(result, duration=15)
    """
    
    def __init__(self, model_size: str = "large-v3", device: Optional[str] = None):
        """
        Initialize transcriber.
        
        Args:
            model_size: Whisper model size (tiny/base/small/medium/large-v3)
            device: Compute device (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        logger.info(f"Loading Whisper model: {model_size} on {self.device}")
        
        try:
            self.model = whisper.load_model(model_size).to(self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    def transcribe(self, 
                   video_path: str,
                   language: Optional[str] = None,
                   task: str = "transcribe") -> Dict:
        """
        Transcribe video and auto-detect language.
        
        Args:
            video_path: Path to video file
            language: Force specific language (None = auto-detect)
            task: "transcribe" or "translate" (to English)
            
        Returns:
            Dictionary with:
                - language: Detected language code
                - full_text: Complete transcription
                - segments: List of timestamped segments
                - duration: Total video duration
        """
        video_path = str(video_path)
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Starting transcription: {video_path}")
        
        try:
            # Run Whisper
            result = self.model.transcribe(
                video_path,
                language=language,
                task=task,
                fp16=(self.device == "cuda"),
                verbose=False
            )
            
            # Extract results
            detected_language = result.get("language", "unknown")
            segments = result.get("segments", [])
            full_text = result.get("text", "").strip()
            
            # Calculate duration
            duration = segments[-1]["end"] if segments else 0
            
            logger.info(f"Transcription complete:")
            logger.info(f"  Language: {detected_language}")
            logger.info(f"  Segments: {len(segments)}")
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"  Text length: {len(full_text)} chars")
            
            return {
                "language": detected_language,
                "full_text": full_text,
                "segments": segments,
                "duration": duration,
                "video_path": video_path
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def find_best_segment(self,
                         transcription: Dict,
                         target_duration: float = 15.0,
                         min_duration: float = 10.0,
                         max_duration: float = 30.0) -> Dict:
        """
        AI-powered selection of the best 15-30 second segment.
        
        Uses multiple factors:
        1. Content density (text per second)
        2. Avoids start/end (usually intros/outros)
        3. Prefers segments with keywords (important content)
        4. Penalizes repetitive content
        
        Args:
            transcription: Output from transcribe()
            target_duration: Ideal segment length
            min_duration: Minimum acceptable
            max_duration: Maximum acceptable
            
        Returns:
            Dictionary with start, end, duration, score, reason
        """
        segments = transcription.get("segments", [])
        total_duration = transcription.get("duration", 0)
        
        if not segments or total_duration == 0:
            raise ValueError("Invalid transcription data")
        
        logger.info(f"Analyzing {len(segments)} segments for best {target_duration}s clip...")
        
        # Keywords indicating important content
        important_keywords = [
            "important", "main", "key", "remember", "note", "careful",
            "hygiene", "clean", "safety", "health", "child", "baby",
            "must", "should", "always", "never", "caution", "warning"
        ]
        
        # Score each possible segment window
        best_score = -1
        best_window = None
        
        # Try different start positions
        step = 1.0  # 1-second steps
        for start in np.arange(0, total_duration - min_duration, step):
            end = min(start + target_duration, total_duration)
            duration = end - start
            
            if duration < min_duration:
                continue
            
            # Get segments in this window
            window_segments = [
                s for s in segments 
                if s["start"] >= start and s["end"] <= end
            ]
            
            if not window_segments:
                continue
            
            # Calculate scores
            
            # 1. Content density (chars per second)
            window_text = " ".join([s["text"] for s in window_segments])
            density = len(window_text) / duration
            
            # 2. Position score (prefer middle, avoid edges)
            center = total_duration / 2
            window_center = (start + end) / 2
            distance_from_center = abs(window_center - center)
            position_score = max(0, 1 - (distance_from_center / center)) if center > 0 else 0
            
            # 3. Keyword score
            text_lower = window_text.lower()
            keyword_count = sum(1 for kw in important_keywords if kw in text_lower)
            keyword_score = min(keyword_count / 3, 1.0)  # Cap at 1
            
            # 4. Repetition penalty (deduplicate words)
            words = text_lower.split()
            unique_words = set(words)
            if len(words) > 0:
                repetition_ratio = 1 - (len(unique_words) / len(words))
                repetition_penalty = max(0, 1 - repetition_ratio * 2)
            else:
                repetition_penalty = 0
            
            # Combined score (weighted)
            score = (
                density * 0.3 +
                position_score * 0.25 +
                keyword_score * 0.25 +
                repetition_penalty * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_window = {
                    "start": float(start),
                    "end": float(end),
                    "duration": float(duration),
                    "score": float(score),
                    "text": window_text[:200] + "..." if len(window_text) > 200 else window_text,
                    "segments": window_segments
                }
        
        if best_window is None:
            # Fallback: middle of video
            middle = total_duration / 2
            start = max(0, middle - target_duration/2)
            best_window = {
                "start": start,
                "end": min(start + target_duration, total_duration),
                "duration": min(target_duration, total_duration),
                "score": 0,
                "text": "Fallback: center of video",
                "segments": []
            }
        
        logger.info(f"Best segment found: {best_window['start']:.1f}s - {best_window['end']:.1f}s")
        
        return best_window
    
    def get_segment_text(self, 
                        transcription: Dict,
                        start: float,
                        end: float) -> str:
        """Get full text for a specific time range."""
        segments = transcription.get("segments", [])
        overlapping = [
            s for s in segments
            if not (s["end"] < start or s["start"] > end)
        ]
        texts = [s["text"].strip() for s in overlapping]
        return " ".join(texts)
    
    def save_transcription(self,
                          transcription: Dict,
                          output_path: str):
        """Save transcription to JSON file."""
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Transcription saved: {output_path}")
    
    def load_transcription(self, input_path: str) -> Dict:
        """Load transcription from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def quick_transcribe(video_path: str, 
                    output_json: Optional[str] = None,
                    model_size: str = "large-v3") -> Dict:
    """Quick transcription without creating class instance."""
    t = Transcriber(model_size=model_size)
    result = t.transcribe(video_path)
    
    if output_json:
        t.save_transcription(result, output_json)
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "transcription.json"
        result = quick_transcribe(video, output)
        print(f"Language: {result['language']}")
        print(f"Duration: {result['duration']:.2f}s")
