# Menial AI

An audio signal processing toolkit for analyzing, isolating, and classifying household sounds. Built for TECHIN 513A — Managing Data And Signal Processing.

**Authors:** Joshua Kirby & Alan Nur

---

## Setup

```bash
cd menial_ai
# python3 -m venv .venv
python3.12 -m venv .venv312
source .venv/bin/activate
pip3 install -r requirements.txt
```

> **Note:** PyTorch requires Python 3.11 or 3.12. If your default Python is 3.14, create the venv with an explicit version: `python3.12 -m venv .venv`

---

## Tools

### 1. Household Sound Monitor (`monitor.py`)

Full monitoring system that listens to the microphone, identifies household sounds, tracks how long each sound persists, and speaks contextual alerts when sounds become concerning.

**Prerequisites:** A trained model in `models/` (see [Training a Model](#training-a-model) below).

```bash
python monitor.py
```

Options:
```
--model-dir PATH      Model directory (default: models/)
--sensitivity FLOAT   Detection confidence threshold, 0.0-1.0 (default: 0.4)
--interval SECS       Seconds between predictions (default: 1.0)
--no-speak            Disable text-to-speech output
```

**Pipeline:** Microphone → NMF Signal Cleaning → Feature Extraction (Mel + MFCC + ZCR + STFT) → CNN Classification → Sound Tracking → Context Engine → Text-to-Speech

The monitor displays a live terminal dashboard showing detected sounds, active durations, and recent alerts. When it detects relevant sounds, it speaks helpful responses — for example, *"I hear chopping. Do you need the next recipe step?"* or *"The water has been running for 5 minutes!"*

### 2. Simple Classifier (`classifier.py`)

Lightweight version — just prints top predictions without tracking or alerts.

```bash
python classifier.py
```

Options:
```
--model-dir PATH   Model directory (default: models/)
--top-k N          Number of predictions to show (default: 3)
--interval SECS    Seconds between predictions (default: 1.0)
```

---

### 3. Web Interface (`server.py`)

Interactive browser-based audio analysis with FFT/STFT visualization, frequency band isolation, and spectral subtraction.

```bash
python server.py
```

Open **http://localhost:5050** in your browser. You can:
- Upload an audio file or choose a synthetic signal preset
- Switch between FFT, STFT, and Fourier Series visualization tabs
- Isolate frequency bands using four modes (Single Band, Multi-Band, Spectral Subtraction, Combined)
- Toggle between original and extracted audio playback with wet/dry mix
- Click "Analyze in Python" to open the advanced isolator GUI

**Keyboard shortcuts:** Space (play/pause), Esc (stop), A (toggle analysis), E (toggle extraction)

---

### 4. Fourier Fundamentals (`fourier_fundamentals.py`)

Generates five educational visualizations explaining FFT and STFT concepts.

```bash
python fourier_fundamentals.py path/to/audio.wav
```

Omit the file path to use a built-in synthetic demo signal. Outputs five PNG figures: time domain, FFT spectrum, windowing effects, STFT spectrogram, and time-frequency tradeoff.

---

### 5. Sound Event Isolator (`isolator.py`)

Interactive GUI for decomposing audio into individual sound events using Watershed, NMF, or CCA algorithms.

```bash
python isolator.py path/to/audio.wav
```

Options:
```
--nfft N        FFT window size (default: 1024)
--overlap N     Overlap percentage (default: 75)
--window TYPE   Window function: hann, hamming, blackman (default: hann)
--maxfreq N     Maximum frequency in Hz (default: 8000)
```

Click on components in the spectrogram to select them. Use sliders to adjust decomposition parameters. Export isolated components as WAV files.

---

## Training a Model

The classifier requires a trained CNN model. Training is done in a Jupyter notebook on Google Colab (free GPU).

### Step 1: Record Custom Training Samples (Optional)

Record 5-second clips for sound categories not covered by ESC-50 (e.g., chopping, fan, faucet):

```bash
python record_samples.py chopping
```

Press Enter to start each recording. Clips are saved to `data/custom/<class_name>/`. Record at least 20 clips per class for best results.

### Step 2: Train on Google Colab

1. Open `training/train_classifier.ipynb` in Google Colab (or use the VS Code Colab extension)
2. If you recorded custom clips, upload the `data/custom/` folder to the Colab environment
3. Run all cells — training takes ~45-90 minutes on a T4 GPU
4. The notebook will:
   - Download the ESC-50 dataset (2,000 clips across 50 sound classes)
   - Merge any custom clips as additional classes
   - Extract 4-channel features per clip: Mel spectrogram, MFCC, ZCR, and STFT magnitude
   - Apply NMF-based signal cleaning as augmentation
   - Train a 4-block CNN with stronger noise augmentation and SpecAugment
   - Compare results with YAMNet (trained on Google's AudioSet)
   - Evaluate accuracy and generate a confusion matrix
   - Export `model.pt`, `labels.json`, and `config.json`

### Step 3: Deploy the Model

Download the three files from Colab and place them in `models/`:

```
models/
├── model.pt       # Trained CNN weights
├── labels.json    # Class names
└── config.json    # Mel spectrogram parameters
```

Then run: `python monitor.py`

---

## Project Structure

```
menial_ai/
├── monitor.py                    # Full monitoring system (NMF clean → CNN → tracking → TTS)
├── classifier.py                 # Simple real-time classifier
├── record_samples.py             # Custom training clip recorder
├── fourier_fundamentals.py       # Educational FFT/STFT visualizer
├── isolator.py                   # Interactive audio decomposition GUI
├── server.py                     # Flask web server
├── fourier_explorer.html         # Web UI markup
├── requirements.txt              # Python dependencies
├── models/                       # Trained model files
├── data/                         # Datasets (gitignored)
├── training/
│   └── train_classifier.ipynb    # Colab training notebook
├── static/
│   ├── css/styles.css
│   └── js/                       # 11 frontend modules
├── audio/                        # Sample audio files
└── documentation/
    └── DOCUMENTATION.md          # Technical architecture docs
```

See [documentation/DOCUMENTATION.md](documentation/DOCUMENTATION.md) for detailed technical documentation.
