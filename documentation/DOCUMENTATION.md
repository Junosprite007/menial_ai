# Menial AI — Project Documentation

**Authors:** Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
**Course:** TECHIN 513A — Managing Data And Signal Processing
**Program:** MS Technology Innovation, University of Washington

---

## Project Overview

Menial AI is an interactive audio signal processing and feature isolation toolkit. It combines educational visualization of Fourier analysis fundamentals with practical audio decomposition tools, delivered through both a Python GUI and a web interface.

### Core Capabilities

- **Educational visualization** of FFT, STFT, and Fourier Series concepts
- **Interactive audio isolation** using multiple decomposition methods
- **Real-time spectrogram visualization** with clickable component selection
- **Web-based interface** for uploading and analyzing audio files
- **Synthetic signal presets** for learning and testing

---

## Project Structure

```
menial_ai/
├── fourier_explorer.html          # Web UI (HTML markup only, ~756 lines)
├── fourier_fundamentals.py        # Educational FFT/STFT visualization tool
├── isolator.py                    # Interactive audio decomposition GUI
├── server.py                      # Flask web server
├── requirements.txt               # Pinned Python dependencies
├── .gitignore                     # Git ignore rules
├── README.md
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

### 1. Fourier Fundamentals (`fourier_fundamentals.py`)

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

### 2. Interactive Isolator (`isolator.py`)

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

### 3. Web Interface (`fourier_explorer.html` + `server.py`)

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
| sounddevice | 0.5.5 | Audio playback |

**Implicit:** Python 3.x, ffmpeg (for audio format conversion in isolator.py)

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

---

## Git Conventions

- **Branch:** Development on `josh`, merge to `main`
- **Ignored files:** `*.wav`, `*.mp3`, `__pycache__/`, `.venv/`, build artifacts
- Generated audio files (e.g., `component_*.wav`, `tmp*.mp3`) are regenerable and excluded from version control
