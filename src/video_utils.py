"""
Video and Audio Processing Utilities

Handles all ffmpeg operations, format conversions, and media processing.
Optimized for Google Colab and local environments.
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing utilities using ffmpeg.
    
    Usage:
        from src.video_utils import VideoProcessor
        vp = VideoProcessor()
        vp.extract_segment("input.mp4", "output.mp4", start=10, duration=15)
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize video processor.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        
        # Verify ffmpeg is available
        try:
            subprocess.run([self.ffmpeg, "-version"], 
                         capture_output=True, check=True)
            logger.info("✅ FFmpeg verified")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Install with: apt-get install ffmpeg")
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict:
        """
        Get video metadata (duration, fps, resolution, codec).
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        video_path = str(video_path)
        
        cmd = [
            self.ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,bit_rate",
            "-show_entries", "format=duration,bit_rate,size",
            "-of", "json",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Parse stream info
            stream = data.get("streams", [{}])[0]
            format_info = data.get("format", {})
            
            # Calculate FPS from fraction (e.g., "25/1" -> 25.0)
            fps_str = stream.get("r_frame_rate", "25/1")
            if "/" in fps_str:
                num, den = map(float, fps_str.split("/"))
                fps = num / den if den != 0 else 25.0
            else:
                fps = float(fps_str)
            
            info = {
                "duration": float(format_info.get("duration", 0)),
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "codec": stream.get("codec_name", "unknown"),
                "video_bitrate": int(stream.get("bit_rate", 0)),
                "file_size": int(format_info.get("size", 0)),
                "format_bitrate": int(format_info.get("bit_rate", 0))
            }
            
            logger.info(f"Video info: {info['width']}x{info['height']} @ {info['fps']}fps, "
                       f"{info['duration']:.2f}s")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def extract_segment(self, 
                       input_path: Union[str, Path],
                       output_path: Union[str, Path],
                       start: float,
                       duration: float,
                       codec: str = "libx264",
                       quality: int = 18) -> Path:
        """
        Extract a segment from video (fast, accurate seek).
        
        Args:
            input_path: Source video path
            output_path: Output segment path
            start: Start time in seconds
            duration: Duration in seconds
            codec: Video codec (libx264, libx265, copy)
            quality: CRF quality (0-51, lower=better, 18=visually lossless)
            
        Returns:
            Path to extracted segment
        """
        input_path = str(input_path)
        output_path = str(output_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Fast seek: input seeking (-ss before -i) for speed
        # But for accuracy with audio, we use output seeking (-ss after -i) for short clips
        cmd = [
            self.ffmpeg,
            "-y",  # Overwrite output
            "-i", input_path,
            "-ss", str(start),           # Start time
            "-t", str(duration),         # Duration
            "-c:v", codec,               # Video codec
            "-preset", "fast",           # Encoding speed
            "-crf", str(quality),        # Quality
            "-c:a", "aac",               # Audio codec
            "-b:a", "192k",              # Audio bitrate
            "-movflags", "+faststart",   # Web optimization
            output_path
        ]
        
        logger.info(f"Extracting segment: {start}s to {start+duration}s")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"✅ Segment extracted: {output_path}")
            return Path(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract segment: {e.stderr}")
    
    def extract_audio(self,
                     video_path: Union[str, Path],
                     output_path: Union[str, Path],
                     sample_rate: int = 22050,
                     mono: bool = True) -> Path:
        """
        Extract audio from video as WAV.
        
        Args:
            video_path: Source video path
            output_path: Output audio path (should end in .wav)
            sample_rate: Target sample rate (22050 for XTTS)
            mono: Convert to mono
            
        Returns:
            Path to extracted audio
        """
        video_path = str(video_path)
        output_path = str(output_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", video_path,
            "-vn",                       # No video
            "-acodec", "pcm_s16le",      # PCM 16-bit little-endian
            "-ar", str(sample_rate),     # Sample rate
        ]
        
        if mono:
            cmd.extend(["-ac", "1"])     # Mono
        else:
            cmd.extend(["-ac", "2"])     # Stereo
            
        cmd.append(output_path)
        
        logger.info(f"Extracting audio at {sample_rate}Hz")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError("Audio file not created")
                
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ Audio extracted: {output_path} ({size_mb:.2f} MB)")
            return Path(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise
    
    def combine_audio_video(self,
                           video_path: Union[str, Path],
                           audio_path: Union[str, Path],
                           output_path: Union[str, Path],
                           copy_video: bool = True) -> Path:
        """
        Replace audio in video with new audio track.
        
        Args:
            video_path: Original video (for video track)
            audio_path: New audio file
            output_path: Output video path
            copy_video: Copy video stream without re-encoding (faster)
            
        Returns:
            Path to output video
        """
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_path = str(output_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", video_path,
            "-i", audio_path,
        ]
        
        if copy_video:
            cmd.extend([
                "-c:v", "copy",          # Copy video (no re-encode)
                "-map", "0:v:0",         # Video from first input
                "-map", "1:a:0",         # Audio from second input
                "-shortest",             # Match shortest duration
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
            ])
        
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ])
        
        logger.info("Combining audio and video")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"✅ Combined video: {output_path}")
            return Path(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise
    
    def resize_video(self,
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    width: Optional[int] = None,
                    height: Optional[int] = None,
                    fps: Optional[int] = None) -> Path:
        """
        Resize video and/or change frame rate.
        
        Args:
            input_path: Source video
            output_path: Output video
            width: New width (None = keep aspect)
            height: New height (None = keep aspect)
            fps: New frame rate (None = keep original)
            
        Returns:
            Path to resized video
        """
        input_path = str(input_path)
        output_path = str(output_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build filter string
        filters = []
        if width or height:
            w = width if width else -1
            h = height if height else -1
            filters.append(f"scale={w}:{h}")
        if fps:
            filters.append(f"fps={fps}")
        
        filter_str = ",".join(filters) if filters else "copy"
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", input_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            output_path
        ]
        
        logger.info(f"Resizing video: {filter_str}")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return Path(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Resize failed: {e.stderr}")
            raise
    
    def get_duration(self, video_path: Union[str, Path]) -> float:
        """Quickly get video duration in seconds."""
        info = self.get_video_info(video_path)
        return info["duration"]
    
    def create_silent_video(self,
                           output_path: Union[str, Path],
                           duration: float,
                           width: int = 640,
                           height: int = 480,
                           fps: int = 25) -> Path:
        """
        Create a silent test video (useful for testing).
        
        Args:
            output_path: Output path
            duration: Video duration
            width: Video width
            height: Video height
            fps: Frame rate
            
        Returns:
            Path to created video
        """
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:d={duration}",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=mono",
            "-shortest",
            "-c:v", "libx264",
            "-t", str(duration),
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return Path(output_path)


class AudioProcessor:
    """
    Audio processing utilities.
    
    Usage:
        from src.video_utils import AudioProcessor
        ap = AudioProcessor()
        ap.time_stretch("input.wav", "output.wav", ratio=1.2)
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path
    
    def time_stretch(self,
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    ratio: float,
                    preserve_pitch: bool = True) -> Path:
        """
        Change audio duration without changing pitch (or with).
        
        Args:
            input_path: Input audio
            output_path: Output audio
            ratio: Time ratio (>1 = slower/longer, <1 = faster/shorter)
            preserve_pitch: Keep original pitch (recommended)
            
        Returns:
            Path to stretched audio
        """
        input_path = str(input_path)
        output_path = str(output_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if preserve_pitch:
            # Use atempo filter (pitch preserved)
            # atempo range is 0.5 to 2.0, chain multiple for extreme values
            tempos = []
            r = ratio
            
            while r > 2.0:
                tempos.append("atempo=2.0")
                r /= 2.0
            while r < 0.5:
                tempos.append("atempo=0.5")
                r /= 0.5
            
            tempos.append(f"atempo={r}")
            filter_str = ",".join(tempos)
        else:
            # Change speed (pitch changes too)
            filter_str = f"asetrate=44100*{ratio},aresample=44100"
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", input_path,
            "-filter:a", filter_str,
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        logger.info(f"Time stretching by ratio: {ratio}")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"✅ Audio stretched: {output_path}")
            return Path(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Stretch failed: {e.stderr}")
            raise
    
    def normalize_audio(self,
                       input_path: Union[str, Path],
                       output_path: Union[str, Path],
                       target_db: float = -14.0) -> Path:
        """
        Normalize audio to target LUFS (loudness).
        
        Args:
            input_path: Input audio
            output_path: Output audio
            target_db: Target loudness in LUFS (-14 for broadcast)
            
        Returns:
            Path to normalized audio
        """
        input_path = str(input_path)
        output_path = str(output_path)
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", input_path,
            "-filter:a", f"loudnorm=I={target_db}:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        logger.info(f"Normalizing audio to {target_db} LUFS")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return Path(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Normalization failed: {e.stderr}")
            raise
    
    def trim_silence(self,
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    silence_threshold: float = -50.0,
                    min_silence_duration: float = 0.5) -> Path:
        """
        Trim silence from start and end of audio.
        
        Args:
            input_path: Input audio
            output_path: Output audio
            silence_threshold: dB threshold for silence
            min_silence_duration: Minimum silence to trim
            
        Returns:
            Path to trimmed audio
        """
        input_path = str(input_path)
        output_path = str(output_path)
        
        # Use silenceremove filter
        filter_str = (f"silenceremove=start_periods=1:start_duration={min_silence_duration}:"
                     f"start_threshold={silence_threshold}dB:"
                     f"stop_periods=1:stop_duration={min_silence_duration}:"
                     f"stop_threshold={silence_threshold}dB")
        
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", input_path,
            "-af", filter_str,
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        logger.info("Trimming silence from audio")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return Path(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Trim failed: {e.stderr}")
            raise


# Convenience functions for quick access
def extract_segment(video_path: Union[str, Path], 
                   output_path: Union[str, Path],
                   start: float, 
                   duration: float) -> Path:
    """Quick extract without creating class instance."""
    vp = VideoProcessor()
    return vp.extract_segment(video_path, output_path, start, duration)

def get_video_info(video_path: Union[str, Path]) -> Dict:
    """Quick info without creating class instance."""
    vp = VideoProcessor()
    return vp.get_video_info(video_path)


if __name__ == "__main__":
    # Test the module
    print("Testing VideoProcessor...")
    vp = VideoProcessor()
    
    # Create a test silent video
    test_video = "/tmp/test_video.mp4"
    vp.create_silent_video(test_video, duration=5.0)
    
    # Get info
    info = vp.get_video_info(test_video)
    print(f"Test video info: {info}")
    
    # Extract segment
    test_segment = "/tmp/test_segment.mp4"
    vp.extract_segment(test_video, test_segment, start=1.0, duration=2.0)
    
    # Extract audio
    test_audio = "/tmp/test_audio.wav"
    vp.extract_audio(test_segment, test_audio)
    
    print("✅ All tests passed!")
