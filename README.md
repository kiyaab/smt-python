# Hand Motion-to-Sentence & Multilingual Translator

> *"Disability is Not Inability."*

An inclusive communication desktop application that tracks hand gestures and body motion in real time using a standard consumer webcam (no specialized hardware required), recognizes sign language glosses, assembles them into grammatical sentences, translates them into six languages (English, Amharic, Spanish, French, Arabic, and Mandarin Chinese), and synthesizes spoken audio via Text-to-Speech (TTS).

Developed under the **STEM Center Innovation Grant** framework.

---

## 🌟 Key Features

- **Zero-Hardware-Cost Tracking:** Runs on standard PC webcams (720p @ 30 FPS) on CPU alone (Intel Core i3 / Ryzen 3, 4 GB RAM minimum).
- **Google MediaPipe Tasks Vision:** Multi-landmark tracking extracting 21 joints per hand (both hands), upper-body pose keypoints, and facial non-manual grammatical cues (eyebrows, mouth).
- **Spatial & Scale Invariance:** Coordinate normalization anchored to dominant wrist and scaled by palm span / shoulder width. Invariant to signer distance and body size.
- **Left- and Right-Handed Support:** Built-in left/right mirror augmentation.
- **Interactive Data Recorder GUI:** Built with PyQt6, featuring a 3-second visual countdown, live skeleton overlay, camera-active privacy indicator, take review, and one-click discard.
- **Multilingual Pipeline (6 Languages):**
  - English (`en` / `eng_Latn`) — Pivot language
  - Amharic (`am` / `amh_Ethi`) — Ethiopic Unicode script
  - Spanish (`es` / `spa_Latn`)
  - French (`fr` / `fra_Latn`)
  - Arabic (`ar` / `arb_Arab`) — Right-to-Left (RTL) layout
  - Mandarin Chinese (`zh-CN` / `zho_Hans`) — Simplified Chinese
- **Synthetic Bootstrapper:** Biomechanically plausible landmark sequence generator for CI/CD and pipeline testing prior to human data collection.
- **100% Free & Open Source:** Zero paid APIs or closed dependencies.

---

## 📁 Repository Structure

```
smt-python/
├── config/
│   ├── default.yaml            # System settings (camera, thresholds, language tags)
│   └── vocabulary.yaml         # 36 core sign glosses + IDLE, categories & translations
├── data/
│   ├── raw/                    # Recorded landmark sequences (.npz)
│   ├── processed/              # Normalized, split datasets
│   └── splits/                 # Signer-independent split manifests
├── models/
│   ├── mediapipe/              # Downloaded MediaPipe .task bundles
│   ├── trained/                # PyTorch checkpoints
│   └── onnx/                   # Optimized ONNX CPU models
├── src/signtranslator/
│   ├── capture/                # Thread-safe video capture & FPS governor
│   ├── tracking/               # MediaPipe Tasks wrappers & coordinate normalizer
│   ├── recognition/            # Dataset, sequence models, segmenter & synthetic generator
│   ├── language/               # Gloss-to-sentence rule & grammar engine
│   ├── translation/            # Multilingual translator (rule + deep-translator + NLLB)
│   ├── speech/                 # Text-to-Speech (gTTS + pyttsx3)
│   ├── ui/                     # Accessible PyQt6 GUI widgets & main window
│   └── tools/                  # PyQt6 Data Recorder tool
├── scripts/
│   ├── download_models.py      # Automated MediaPipe task asset downloader
│   ├── generate_synthetic_data.py # Bootstrap landmark sequence generator
│   └── benchmark_tracking.py   # Real CPU latency and FPS benchmark
├── tests/                      # Automated unit test suite
├── pyproject.toml              # Build & package configuration
├── requirements.txt            # Dependency manifest
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Standard webcam
- Windows 10/11, macOS, or Ubuntu 20.04+

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/kiyaab/smt-python.git
cd smt-python

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e . --no-deps
```

### 3. Download MediaPipe Models
Download the official Google MediaPipe Tasks vision model bundles into `models/mediapipe/`:
```bash
python scripts/download_models.py
```

---

## 🎮 Usage & Tools

### Launch the Data Recorder GUI
Collect custom sign language takes using your webcam:
```bash
python -m signtranslator.tools.recorder
```
- **Controls:**
  - Select a sign gloss from the dropdown.
  - Press `Spacebar` (or click **Record Take**) to start the 3-second countdown.
  - Perform the sign. The sequence will be saved to `data/raw/{gloss}_{signer}_{timestamp}.npz`.
  - Press `Delete` (or click **Discard Last Take**) to delete the last take.

### Generate Synthetic Test Sequences
Generate bootstrap landmark sequences for automated testing:
```bash
python scripts/generate_synthetic_data.py --samples-per-class 15
```

### Run the CPU Performance Benchmark
Measure landmark tracking latency and throughput on CPU:
```bash
python scripts/benchmark_tracking.py
```

### Run the Test Suite
Run all unit tests:
```bash
pytest -v
```

---

## 🛡️ Privacy & Accessibility

- **Local-First Processing:** Video frames are processed strictly in memory on the local machine and never stored or transmitted to external servers. Only compact 3D landmark coordinates (`.npz`) are stored when explicitly recording training takes.
- **Privacy Indicator:** Prominent visual badge indicates when the webcam is active.
- **Community Disclaimer:** This system is an assistive communication aid for inclusive daily interaction and is **not** a certified replacement for qualified human sign language interpreters in legal, medical, or life-critical scenarios.

---

## 📄 License
MIT License. Open-source for academic, humanitarian, and inclusive communication development.
