# Menial AI — Project Documentation

**Authors:** Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
**Course:** TECHIN 513A — Managing Data And Signal Processing
**Program:** MS Technology Innovation, University of Washington

---

## Project Overview

Menial AI is an interactive audio signal processing and feature isolation toolkit. It combines educational visualization of Fourier analysis fundamentals with practical audio decomposition tools, delivered through both a Python GUI and a web interface.

### Core Capabilities

- **Real-time sound monitoring** with contextual voice alerts using a custom-trained CNN
- **Multi-feature classification** using MFCC, STFT, ZCR, and NMF signal processing techniques
- **Educational visualization** of FFT, STFT, and Fourier Series concepts
- **Interactive audio isolation** using multiple decomposition methods
- **Real-time spectrogram visualization** with clickable component selection
- **Web-based interface** for uploading and analyzing audio files
- **Synthetic signal presets** for learning and testing

---

## Project Structure

```
menial_ai/
├── monitor.py                     # Full monitoring system (clean → classify → track → speak)
├── classifier.py                  # Simple real-time classifier
├── record_samples.py              # Helper to record custom training clips
├── fourier_explorer.html          # Web UI (HTML markup only, ~756 lines)
├── fourier_fundamentals.py        # Educational FFT/STFT visualization tool
├── isolator.py                    # Interactive audio decomposition GUI
├── server.py                      # Flask web server
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── README.md
│
├── models/                        # Trained model weights + config
│   ├── model.pt                   # CNN state dict
│   ├── labels.json                # Class index → name mapping
│   └── config.json                # Mel spectrogram parameters
│
├── training/
│   └── train_classifier.ipynb     # Google Colab training notebook
│
├── data/                          # Datasets (gitignored)
│   ├── ESC-50/                    # ESC-50 dataset (downloaded by notebook)
│   └── custom/                    # Self-recorded clips by class folder
│
├── static/
│   ├── css/
│   │   └── styles.css             # All UI styles + Google Fonts import
│   └── js/
│       ├── lessons.js             # $ alias, lesson content, L(), togLP()
│       ├── dsp.js                 # FFT, STFT, filtering, spectral subtraction
│       ├── generators.js          # Synthetic signal presets
│       ├── colormap.js            # Magma colormap, band colors
│       ├── state.js               # Global state object S
│       ├── extraction.js          # recomputeExtraction, getPlaySig
│       ├── playback.js            # Web Audio API, transport controls
│       ├── ui.js                  # Mode switching, band UI, inline editing
│       ├── rendering.js           # All canvas drawing + parameter display
│       ├── io.js                  # File loading, presets, Python bridge
│       └── main.js                # Event listeners, initialization
│
├── audio/                         # Sample audio files
├── documentation/                 # Project docs, timeline, proposal
└── .venv/                         # Python virtual environment
```

---

## Components

### 1. Household Sound Monitor (`monitor.py`)

Full monitoring system implementing the project proposal's pipeline: Microphone → Signal Cleaning → AI Model → Identify Sound → Robot Aware Context & Speaks.

**Architecture:**
```
HouseholdMonitor (orchestrator)
├── SignalCleaner      — NMF-based noise reduction (4 components, drop lowest-energy)
├── FeatureExtractor   — 4-channel: Mel spectrogram + MFCC + ZCR + STFT magnitude
├── SoundTracker       — State machine: IDLE → DETECTED → ACTIVE → FADING → IDLE
├── ContextEngine      — Rule engine mapping (sound, event, duration) → response text
├── Speaker            — macOS `say` / pyttsx3 TTS with 30-second debounce
└── Display            — ANSI terminal dashboard with colors, durations, alert history
```

**Pipeline per cycle (~60-130ms):**
```
1. Copy 5-second ring buffer
2. SignalCleaner.clean()     — STFT → NMF(4 components) → drop noise → iSTFT
3. FeatureExtractor.extract() — 4-channel tensor (4, 128, T)
4. CNN forward pass           — top-5 predictions with softmax
5. SoundTracker.update()      — state transitions, duration milestones
6. ContextEngine.evaluate()   — generate response text
7. Speaker.speak()            — non-blocking TTS
8. Display.render()           — terminal dashboard
```

**Context rules (examples):**

| Sound | Trigger | Response |
|-------|---------|----------|
| chopping | started | "I hear chopping. Do you need the next recipe step?" |
| water | 2 min | "The water has been running for 2 minutes. Did you forget to turn it off?" |
| boiling | started | "I hear boiling water. Would you like me to set a timer?" |
| boiling | 10 min | "Warning: boiling for 10 minutes. You might want to check the stove." |
| smoke detector | started | "ALERT: Smoke detector going off! Please check immediately." |

### 2. Simple Classifier (`classifier.py`)

Lightweight version — classifies household sounds and prints predictions without tracking or alerts.

**Architecture:**
```
Microphone → sounddevice.InputStream → 5-second ring buffer
                                            ↓ (every 1 second)
                                       Mel Spectrogram
                                       (n_fft=2048, hop=512, 128 mels)
                                            ↓
                                       HouseholdSoundCNN
                                       (4-block CNN, ~150K params)
                                            ↓
                                       softmax → top-K predictions → terminal
```

**Model — `HouseholdSoundCNN`:**

| Layer | Output Shape |
|-------|-------------|
| Conv2d(1→16, 3x3) + BN + ReLU + MaxPool | (16, 64, 215) |
| Conv2d(16→32, 3x3) + BN + ReLU + MaxPool | (32, 32, 107) |
| Conv2d(32→64, 3x3) + BN + ReLU + MaxPool | (64, 16, 53) |
| Conv2d(64→128, 3x3) + BN + ReLU + MaxPool | (128, 8, 26) |
| AdaptiveAvgPool2d(1) | (128, 1, 1) |
| Flatten + Dropout(0.3) + Linear(128→64) + ReLU | (64,) |
| Dropout(0.3) + Linear(64→num_classes) | (num_classes,) |

The model definition is duplicated in both `classifier.py` and `training/train_classifier.ipynb` and must stay in sync.

**Required files in `models/`:**

| File | Contents |
|------|----------|
| `model.pt` | PyTorch state dict (trained weights) |
| `labels.json` | `{"num_classes": N, "labels": ["class1", ...]}` |
| `config.json` | Mel spectrogram parameters (must match training) |

### 3. Training Pipeline (`training/train_classifier.ipynb`)

Google Colab notebook that trains the CNN classifier.

**Dataset:** ESC-50 (2,000 five-second clips, 50 environmental sound classes) plus optional self-recorded clips in `data/custom/`. Custom clips are assigned to training folds automatically.

**Feature extraction (4-channel input):**
```
Audio (any format) → Resample to 44100 Hz → Mono → Pad/trim to 5s
    → Channel 0: MelSpectrogram → AmplitudeToDB           (128, T)
    → Channel 1: MFCC (40 coefficients) → resize to 128   (128, T)
    → Channel 2: Zero-Crossing Rate → broadcast            (128, T)
    → Channel 3: STFT magnitude → resize to 128            (128, T)
    → Stack → Output: (4, 128, T) tensor
```

**Data augmentation (training only):**
- Time shift (random roll up to ±1 second)
- Variable Gaussian noise (σ=0.001–0.02)
- Background noise mixing (ambient ESC-50 clips at 5-20% volume)
- Volume variation (gain 0.5x–1.5x)
- NMF denoising (applied to 20% of clips)
- SpecAugment: frequency masking (25 bins) + time masking (50 frames)

**Training config:**
- Optimizer: Adam (lr=1e-3) with ReduceLROnPlateau (factor=0.5, patience=5)
- Loss: CrossEntropyLoss
- Split: ESC-50 folds 1-3 = train, fold 4 = validation, fold 5 = test
- Epochs: 100 (with early stopping via best-model checkpoint)

**AudioSet comparison:** YAMNet (trained on AudioSet's 2M+ clips) is loaded via TensorFlow Hub and compared against the custom CNN on test samples.

**Outputs:** `model.pt`, `labels.json`, `config.json` (with `n_channels: 4`) → download to `models/`

### 4. Sample Recorder (`record_samples.py`)

Records 5-second microphone clips for custom training categories not covered by ESC-50.

Saves clips as 16-bit WAV at 44100 Hz to `data/custom/<class_name>/<class_name>_001.wav` with auto-incrementing filenames.

### 5. Fourier Fundamentals (`fourier_fundamentals.py`)

A standalone educational script (~650 lines) that generates five progressive figures teaching Fourier analysis concepts:

1. **Time Domain** — Raw waveform visualization with sample-level zoom
2. **FFT Spectrum** — Frequency domain representation in linear and dB scales
3. **FFT Refinements** — Windowing and zero-padding effects
4. **STFT Spectrogram** — Time-frequency representation with labeled parameters
5. **Time-Frequency Tradeoff** — How different `n_fft` values affect resolution

**Usage:**
```bash
python fourier_fundamentals.py <path_to_wav_file>
```

Handles multiple audio formats (mono/stereo conversion, various bit depths). Falls back to a synthetic demo signal if no file is provided.

### 6. Interactive Isolator (`isolator.py`)

A matplotlib-based GUI (~877 lines) implementing three audio decomposition algorithms:

| Method | Approach | Best For |
|--------|----------|----------|
| **Watershed** | Peak-based flood-fill segmentation | Isolated, distinct sound events |
| **NMF** | Non-negative matrix factorization (custom numpy) | Overlapping sound sources |
| **CCA** | Connected component analysis (threshold + morphology) | Fast, simple blob detection |

**Features:**
- Three-panel display: waveform, spectrogram with overlay, isolated component
- Click to select components, real-time parameter adjustment via sliders
- Audio playback via `sounddevice`
- Export isolated components as WAV files
- Configurable STFT parameters (n_fft, overlap, window type, max frequency)

**Data Processing Pipeline:**
```
Audio Load → STFT → Magnitude + Phase →
Decomposition (Watershed/NMF/CCA) →
Mask Creation → Audio Reconstruction (Magnitude × Mask + Phase) →
Inverse STFT → Output Audio
```

### 7. Web Interface (`fourier_explorer.html` + `server.py`)

A web-based audio analysis tool with a Flask backend.

#### Frontend Features

- **4 isolation modes:** Single Band, Multi-Band, Spectral Subtraction, Combined
- **Custom Cooley-Tukey FFT** running entirely client-side in JavaScript
- **Butterworth bandpass filtering** in the frequency domain
- **Real-time playback** with wet/dry mix, live spectrum visualization
- **5 synthetic signal presets:** Tones, Chirp, Impulses, C Major Chord, Square Wave
- **3 visualization tabs:** FFT spectrum, STFT spectrogram, Fourier Series
- **50+ educational lessons** in a collapsible panel
- **Keyboard shortcuts:** Space (play/pause), Esc (stop), A (analysis toggle), E (extracted toggle)

#### Backend (`server.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves `fourier_explorer.html` |
| `/analyze` | POST | Accepts audio upload, launches isolator GUI |

**Configuration:** Port 5050, file type validation (WAV, MP3, OGG, FLAC, M4A), 50MB upload limit.

**Usage:**
```bash
python server.py
# Open http://localhost:5050
```

---

## Frontend Architecture

The web frontend was split from a single 4,442-line HTML file into 13 modular files for maintainability:

### Script Load Order

Scripts must load in this order due to cross-file dependencies:

```html
<script src="/static/js/lessons.js"></script>    <!-- $ alias, lessons -->
<script src="/static/js/dsp.js"></script>         <!-- pure math, no deps -->
<script src="/static/js/generators.js"></script>  <!-- signal generators -->
<script src="/static/js/colormap.js"></script>    <!-- colormap data -->
<script src="/static/js/state.js"></script>       <!-- S object -->
<script src="/static/js/extraction.js"></script>  <!-- uses S, dsp -->
<script src="/static/js/playback.js"></script>    <!-- uses S, $ -->
<script src="/static/js/ui.js"></script>          <!-- uses S, $, extraction -->
<script src="/static/js/rendering.js"></script>   <!-- uses everything above -->
<script src="/static/js/io.js"></script>          <!-- uses S, $, generators -->
<script src="/static/js/main.js"></script>        <!-- event binding, runs last -->
```

### Module Responsibilities

| Module | Lines | Key Symbols |
|--------|-------|-------------|
| `lessons.js` | 200 | `$`, `LS`, `EDU`, `L()`, `togLP()` |
| `dsp.js` | 361 | `gW`, `fI`, `ifI`, `fftSlice`, `fullFFT`, `compSTFT`, `bandpassFilter`, `butterworthGain`, `smoothBandpass`, `multiBandFilter`, `captureNoise`, `spectralSubtract`, `compFS`, `reconstructFS` |
| `generators.js` | 63 | `gTones`, `gChirp`, `gImp`, `gChord`, `gSq` |
| `colormap.js` | 66 | `MG`, `mc()`, `HC` |
| `state.js` | 55 | `S` (global state object) |
| `extraction.js` | 110 | `recomputeExtraction`, `getPlaySig` |
| `playback.js` | 127 | `ensCtx`, `buildBuf`, `togPlay`, `startA`, `pauseA`, `stopA`, `curT`, `seekTo`, `fmtT`, `updTr`, `togExtPlay` |
| `ui.js` | 144 | `switchMode`, `addBand`, `removeBand`, `renderBands`, `makeEditable`, `togOverlay`, `togAn` |
| `rendering.js` | 849 | `rAll`, `rFr`, `gC`, `dWave`, `dFFT`, `dSTFT`, `dFS`, `dLive`, `dOff`, `updP`, `pk`, `startAL`, `stopAL` |
| `io.js` | 203 | `ldP`, `ldFile`, `updFSMax`, `sendToPython`, `encodeWAV` |
| `main.js` | 704 | `init()`, all event listeners, keyboard shortcuts, startup |

### Global State (`S` object)

All application state lives in a single `S` object defined in `state.js`. Key properties:

- **Signal:** `sig` (Float64Array), `sr` (sample rate), `lbl` (label), `dur` (duration)
- **Playback:** `playing`, `pOff` (offset), `aCtx` (AudioContext), `playExt` (extracted mode)
- **Analysis:** `tab`, `wt` (window type), `nfft`, `olP` (overlap %), `zpR` (zero-pad ratio), `dynR` (dynamic range), `mf` (max frequency)
- **Extraction:** `extSig`, `isoMode`, `bLo`/`bHi` (band cutoffs), `rolloff`, `wetDry`, `bands` (multi-band array)
- **Spectral Subtraction:** `noiseProfile`, `overSub` (alpha), `specFloor` (beta)
- **Fourier Series:** `fsW` (window size), `fsH` (harmonics), `fsOff` (offset)
- **Selection:** `selStart`, `selEnd`, `hasSelection`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.10.8 | Visualization and interactive GUI (isolator) |
| numpy | 2.4.2 | Numerical computing, array operations |
| scipy | 1.17.0 | Signal processing (STFT, ISTFT, morphological ops) |
| flask | 3.1.3 | Web server framework |
| sounddevice | 0.5.5 | Audio playback and microphone input |
| torch | ≥2.0 | CNN model definition and inference |
| torchaudio | ≥2.0 | Mel spectrogram transforms, audio loading |

**Implicit:** Python 3.11–3.12 (for PyTorch compatibility), ffmpeg (for audio format conversion in isolator.py)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Key DSP Concepts Implemented

| Concept | Implementation | Location |
|---------|---------------|----------|
| FFT (Cooley-Tukey) | `fI()` / `ifI()` — radix-2 DIT with bit-reversal | `dsp.js` |
| STFT | `compSTFT()` — windowed, overlapping FFT frames | `dsp.js` |
| Butterworth Filter | `butterworthGain()` / `smoothBandpass()` — frequency-domain gain curve | `dsp.js` |
| Multi-Band Filtering | `multiBandFilter()` — max-gain union across bands | `dsp.js` |
| Spectral Subtraction | `spectralSubtract()` — noise profile estimation and removal | `dsp.js` |
| NMF | Custom multiplicative update rules — `V ≈ W @ H` | `isolator.py` |
| Watershed Segmentation | Peak detection + iterative flood-fill on spectrogram | `isolator.py` |
| Connected Component Analysis | Thresholding + morphological ops + labeling | `isolator.py` |
| Window Functions | Rectangular, Hann, Hamming, Blackman | `dsp.js`, `fourier_fundamentals.py` |
| Phase Reconstruction | Magnitude masking with original phase preservation | `isolator.py` |
| Fourier Series | `compFS()` / `reconstructFS()` — harmonic decomposition | `dsp.js` |
| Mel Spectrogram | STFT → mel filterbank → log scaling for CNN input | `classifier.py`, `train_classifier.ipynb` |
| CNN Classification | 4-block ConvNet with global average pooling | `classifier.py`, `train_classifier.ipynb` |
| SpecAugment | Frequency/time masking for training augmentation | `train_classifier.ipynb` |

---

## Git Conventions

- **Branch:** Development on `josh`, merge to `main`
- **Ignored files:** `*.wav`, `*.mp3`, `__pycache__/`, `.venv/`, build artifacts
- Generated audio files (e.g., `component_*.wav`, `tmp*.mp3`) are regenerable and excluded from version control
