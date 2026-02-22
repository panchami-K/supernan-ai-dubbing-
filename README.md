# 🎬 Supernan Golden 15 Seconds - AI Dubbing Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade multilingual AI video dubbing with intelligent segment selection**

## 🌟 Features

- 🎯 **Auto Language Detection**: Supports 99+ languages via Whisper
- 🧠 **AI-Powered Segment Selection**: Automatically finds the best 15-30 seconds
- 🗣️ **Voice Cloning**: Preserves original speaker's tone and style
- 👄 **Perfect Lip Sync**: Advanced Wav2Lip integration
- ✨ **Face Enhancement**: GFPGAN for professional video quality
- 🌐 **Any-to-Any Translation**: Kannada→Hindi, English→Hindi, etc.
- 🖥️ **User-Friendly UI**: Streamlit web interface

## 📁 Repository Structure
supernan-ai-dubbing-/
├── src/                    # Core pipeline modules
│   ├── config.py          # Configuration management
│   ├── transcribe.py      # Speech-to-text (Whisper)
│   ├── segment_analyzer.py # Best segment selection
│   ├── translate.py       # Neural machine translation
│   ├── voice_clone.py     # XTTS voice cloning
│   ├── lip_sync.py        # Wav2Lip integration
│   ├── face_restore.py    # GFPGAN face enhancement
│   ├── video_utils.py     # Video processing utilities
│   └── pipeline.py        # Main orchestrator
├── web_app/               # Streamlit UI
│   └── app.py
├── data/                  # Input/output videos
├── dub_video.py          # CLI entry point
└── requirements.txt

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/panchami-K/supernan-ai-dubbing-.git
cd supernan-ai-dubbing-

# Install dependencies
pip install -r requirements.txt

# Download models (one-time)
python scripts/download_models.py

CLI Usage
# Full pipeline on video
python dub_video.py --input video.mp4 --target-lang hi --output dubbed.mp4

# With custom segment selection
python dub_video.py --input video.mp4 --start 45 --end 60 --target-lang hi

Web UI
bash
streamlit run web_app/app.py

🏗️ Pipeline Architecture
plain
Copy
Input Video (Any Language)
    ↓
[Transcribe] → Whisper (Auto-detect language)
    ↓
[Analyze] → AI selects best 15-30s segment
    ↓
[Translate] → IndicTrans2 / NLLB
    ↓
[Voice Clone] → XTTS (preserves speaker tone)
    ↓
[Lip Sync] → Wav2Lip (matches lips to audio)
    ↓
[Enhance] → GFPGAN (face restoration)
    ↓
Output Video (Target Language)


👤 Author
Panchami K
Email: panchamik12345@gmail.com
GitHub: @panchami-K
📜 License
MIT License - see LICENSE file
