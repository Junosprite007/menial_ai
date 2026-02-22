"""
============================================================================
FOURIER TRANSFORM FUNDAMENTALS FOR AUDIO SIGNAL PROCESSING
============================================================================
This file was created with the assistance of Claude Opus 4.6 LLM

Authors: Joshua Kirby & Alan Nur
Purpose: Understand every parameter of the FFT and STFT by applying them
         to a real .wav file, with full visualizations.

How to run:
    python fourier_fundamentals.py your_audio.wav

This script produces 5 figures that build understanding step by step:
    1. The raw waveform (time domain)
    2. The FFT magnitude spectrum (frequency domain)
    3. FFT parameter exploration (zero-padding, windowing)
    4. The STFT spectrogram with labeled parameters
    5. STFT parameter comparison (window size tradeoff)
============================================================================
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, stft

# ──────────────────────────────────────────────────────────────────────────
# 0. LOAD THE .WAV FILE
# ──────────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python fourier_fundamentals.py <path_to_wav_file>")
    print("No file provided — generating a synthetic demo signal instead.\n")

    # Synthetic signal: 3 sine waves + noise (so you can verify the FFT works)
    fs = 44100  # Sample rate in Hz
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Three known frequencies you'll be able to spot in the FFT output
    signal = (
        0.7 * np.sin(2 * np.pi * 440 * t)  # A4 note (440 Hz)
        + 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone
        + 0.3 * np.sin(2 * np.pi * 2500 * t)  # 2.5 kHz tone
    )
    signal += 0.05 * np.random.randn(len(signal))  # small noise floor
    source_label = "Synthetic (440 Hz + 1 kHz + 2.5 kHz)"
else:
    filepath = sys.argv[1]
    fs, raw_data = wavfile.read(filepath)

    # Handle stereo → mono conversion
    if raw_data.ndim == 2:
        raw_data = raw_data.mean(axis=1)

    # Normalize to [-1, 1] float range regardless of bit depth
    if raw_data.dtype == np.int16:
        signal = raw_data.astype(np.float64) / 32768.0
    elif raw_data.dtype == np.int32:
        signal = raw_data.astype(np.float64) / 2147483648.0
    elif raw_data.dtype == np.float32 or raw_data.dtype == np.float64:
        signal = raw_data.astype(np.float64)
    else:
        signal = raw_data.astype(np.float64) / np.max(np.abs(raw_data))

    t = np.arange(len(signal)) / fs
    source_label = filepath

print(f"Source:       {source_label}")
print(f"Sample Rate:  {fs} Hz")
print(f"Duration:     {len(signal) / fs:.2f} seconds")
print(f"Num Samples:  {len(signal)}")
print()


# ======================================================================== #
#                                                                          #
#                    PART 1 — THE TIME DOMAIN                              #
#                                                                          #
# ======================================================================== #

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  KEY PARAMETER: SAMPLE RATE (fs)                                        ║
║                                                                          ║
║  What it is:  How many times per second the microphone measured the      ║
║               air pressure. Each measurement = one "sample."             ║
║                                                                          ║
║  Unit:        Hertz (Hz) = samples per second                            ║
║                                                                          ║
║  Why it matters:                                                         ║
║    The Nyquist theorem says you can only faithfully capture frequencies   ║
║    up to fs/2. For fs=44100, that's 22050 Hz — just above human hearing. ║
║    Any frequency above fs/2 in the original sound "folds back" into      ║
║    lower frequencies (aliasing), corrupting your data.                    ║
║                                                                          ║
║  For your project:                                                       ║
║    Household sounds rarely exceed ~8 kHz, so fs=16000 is common for      ║
║    classification tasks (YAMNet uses 16 kHz). Higher fs gives you more   ║
║    frequency range but larger files and slower processing.                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

fig1, axes1 = plt.subplots(2, 1, figsize=(14, 6))
fig1.suptitle(
    "FIGURE 1 — Time Domain: Your Raw Audio Signal", fontsize=14, fontweight="bold"
)

# Full waveform
axes1[0].plot(t, signal, linewidth=0.3, color="#2563eb")
axes1[0].set_xlabel("Time (seconds)")
axes1[0].set_ylabel("Amplitude")
axes1[0].set_title(
    f"Full Waveform  |  fs = {fs} Hz  |  N = {len(signal)} samples  |  Duration = {len(signal) / fs:.2f}s"
)
axes1[0].grid(True, alpha=0.3)

# Zoomed view — show individual samples
zoom_samples = 200
axes1[1].plot(
    t[:zoom_samples],
    signal[:zoom_samples],
    "o-",
    markersize=2,
    linewidth=0.8,
    color="#dc2626",
)
axes1[1].set_xlabel("Time (seconds)")
axes1[1].set_ylabel("Amplitude")
axes1[1].set_title(
    f"Zoomed: First {zoom_samples} Samples  |  Sample spacing = {1 / fs * 1000:.4f} ms"
)
axes1[1].grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("01_time_domain.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 01_time_domain.png")


# ======================================================================== #
#                                                                          #
#                    PART 2 — THE FFT (Fast Fourier Transform)             #
#                                                                          #
# ======================================================================== #

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WHAT THE FFT DOES                                                       ║
║                                                                          ║
║  The FFT takes your N time-domain samples and decomposes them into       ║
║  N/2 + 1 frequency "bins." Each bin tells you:                           ║
║    - MAGNITUDE: how loud that frequency is in the signal                 ║
║    - PHASE: the time offset of that frequency's sine wave                ║
║                                                                          ║
║  Think of it as: "If I had to rebuild this entire sound using only       ║
║  pure sine waves, how loud would each sine wave need to be?"             ║
║                                                                          ║
║  ─── ALL THE FFT PARAMETERS ───                                         ║
║                                                                          ║
║  N (number of input samples):                                            ║
║    • Determines frequency resolution: Δf = fs / N                        ║
║    • More samples → finer frequency detail (narrower bins)               ║
║    • Fewer samples → coarser bins but faster computation                  ║
║                                                                          ║
║  fs (sample rate):                                                       ║
║    • Sets the maximum detectable frequency: f_max = fs / 2 (Nyquist)     ║
║    • Determines the physical frequency each bin maps to                   ║
║                                                                          ║
║  Frequency bins (the output x-axis):                                     ║
║    • bin[k] corresponds to frequency: f_k = k × (fs / N)                ║
║    • You get N/2 + 1 unique bins (0 Hz to fs/2)                          ║
║    • The other N/2 bins are mirror images (negative frequencies)          ║
║                                                                          ║
║  Magnitude spectrum:                                                     ║
║    • |FFT[k]| = amplitude of frequency bin k                             ║
║    • Often converted to decibels: 20 × log10(|FFT[k]|)                   ║
║                                                                          ║
║  Phase spectrum:                                                         ║
║    • angle(FFT[k]) = phase offset in radians                             ║
║    • Critical for reconstruction, often ignored for classification        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

N = len(signal)

# Compute the FFT
fft_result = np.fft.rfft(signal)  # rfft = "real FFT" — only positive freqs
fft_magnitude = np.abs(fft_result)  # How loud each frequency is
fft_phase = np.angle(fft_result)  # Phase offset of each frequency
fft_freqs = np.fft.rfftfreq(N, d=1 / fs)  # Map bin index → physical frequency (Hz)

# Convert magnitude to decibels (log scale — how humans perceive loudness)
fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)  # +1e-10 avoids log(0)

# Print the key derived parameters
freq_resolution = fs / N
nyquist = fs / 2
num_bins = len(fft_freqs)

print(f"\n{'=' * 60}")
print(f"  FFT PARAMETERS")
print(f"{'=' * 60}")
print(f"  Input samples (N):        {N}")
print(f"  Sample rate (fs):         {fs} Hz")
print(f"  Nyquist frequency:        {nyquist} Hz")
print(f"  Frequency resolution Δf:  {freq_resolution:.4f} Hz")
print(f"  Number of freq bins:      {num_bins}  (= N/2 + 1)")
print(f"  Bin spacing:              {freq_resolution:.4f} Hz per bin")
print(f"{'=' * 60}\n")

# ── FIGURE 2: FFT Magnitude Spectrum ──

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7))
fig2.suptitle(
    "FIGURE 2 — FFT: Frequency Domain Representation", fontsize=14, fontweight="bold"
)

# Linear scale
axes2[0].plot(fft_freqs, fft_magnitude, linewidth=0.4, color="#7c3aed")
axes2[0].set_xlabel("Frequency (Hz)")
axes2[0].set_ylabel("Magnitude (linear)")
axes2[0].set_title(
    f"FFT Magnitude (Linear Scale)  |  Δf = {freq_resolution:.4f} Hz  |  "
    f"{num_bins} bins  |  Nyquist = {nyquist} Hz"
)
axes2[0].set_xlim(0, min(nyquist, 10000))  # Focus on audible range
axes2[0].grid(True, alpha=0.3)

# Decibel scale (more useful — matches human perception)
axes2[1].plot(fft_freqs, fft_magnitude_db, linewidth=0.4, color="#059669")
axes2[1].set_xlabel("Frequency (Hz)")
axes2[1].set_ylabel("Magnitude (dB)")
axes2[1].set_title("FFT Magnitude (Decibel Scale)  |  dB = 20 × log₁₀(|FFT|)")
axes2[1].set_xlim(0, min(nyquist, 10000))
axes2[1].grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("02_fft_spectrum.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 02_fft_spectrum.png")


# ======================================================================== #
#                                                                          #
#         PART 3 — FFT PARAMETER EXPLORATION                              #
#         (Windowing and Zero-Padding)                                     #
#                                                                          #
# ======================================================================== #

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  TWO CRITICAL FFT REFINEMENTS                                           ║
║                                                                          ║
║  1. WINDOWING                                                            ║
║     Problem: The FFT assumes your signal repeats forever. But your       ║
║     audio chunk has sharp edges (it starts and stops abruptly). These    ║
║     discontinuities create fake high-frequency artifacts = "spectral     ║
║     leakage."                                                            ║
║                                                                          ║
║     Solution: Multiply your signal by a "window function" that gently    ║
║     tapers the edges to zero. Common windows:                            ║
║       • Rectangular (no window) — maximum leakage, sharpest peaks        ║
║       • Hann — good general purpose, smooth taper                        ║
║       • Hamming — slightly less taper at edges than Hann                  ║
║       • Blackman — strongest sidelobe suppression, widest main lobe      ║
║                                                                          ║
║     Tradeoff: Better leakage suppression = wider main lobe = less       ║
║     ability to distinguish two close frequencies.                        ║
║                                                                          ║
║  2. ZERO-PADDING                                                         ║
║     Adding zeros to the end of your signal before the FFT.               ║
║     This does NOT add new frequency information!                         ║
║     It interpolates between existing bins, making the spectrum smoother.  ║
║     Think of it as "increasing the pixel count" of your frequency plot.  ║
║                                                                          ║
║     n_fft > N:  the signal is zero-padded to length n_fft               ║
║     New Δf = fs / n_fft  (appears finer, but not truly more resolving)  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# Take a short chunk of the signal for clearer demonstration
chunk_duration = 0.05  # 50 milliseconds
chunk_samples = int(fs * chunk_duration)
chunk = signal[:chunk_samples]
chunk_t = t[:chunk_samples]

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle(
    "FIGURE 3 — FFT Refinements: Windowing & Zero-Padding",
    fontsize=14,
    fontweight="bold",
)

# ── Panel A: Compare window functions on the chunk ──
windows = {
    "Rectangular (none)": np.ones(chunk_samples),
    "Hann": get_window("hann", chunk_samples),
    "Hamming": get_window("hamming", chunk_samples),
    "Blackman": get_window("blackman", chunk_samples),
}
colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]

for (name, win), color in zip(windows.items(), colors):
    axes3[0, 0].plot(chunk_t * 1000, win, label=name, linewidth=1.5, color=color)
axes3[0, 0].set_xlabel("Time (ms)")
axes3[0, 0].set_ylabel("Window amplitude")
axes3[0, 0].set_title("Window Functions (shape)")
axes3[0, 0].legend(fontsize=8)
axes3[0, 0].grid(True, alpha=0.3)

# ── Panel B: FFT of windowed chunk (effect on spectrum) ──
for (name, win), color in zip(windows.items(), colors):
    windowed = chunk * win
    fft_w = np.fft.rfft(windowed)
    freqs_w = np.fft.rfftfreq(chunk_samples, d=1 / fs)
    mag_db = 20 * np.log10(np.abs(fft_w) + 1e-10)
    axes3[0, 1].plot(freqs_w, mag_db, label=name, linewidth=0.8, color=color, alpha=0.8)

axes3[0, 1].set_xlabel("Frequency (Hz)")
axes3[0, 1].set_ylabel("Magnitude (dB)")
axes3[0, 1].set_title(
    f"FFT of Windowed Chunk ({chunk_samples} samples = {chunk_duration * 1000:.0f} ms)"
)
axes3[0, 1].set_xlim(0, min(nyquist, 8000))
axes3[0, 1].legend(fontsize=8)
axes3[0, 1].grid(True, alpha=0.3)

# ── Panel C: Zero-padding demonstration ──
hann_win = get_window("hann", chunk_samples)
windowed_chunk = chunk * hann_win

nfft_values = [chunk_samples, chunk_samples * 2, chunk_samples * 4, chunk_samples * 8]
zp_colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]

for nfft, color in zip(nfft_values, zp_colors):
    padded = np.zeros(nfft)
    padded[:chunk_samples] = windowed_chunk
    fft_zp = np.fft.rfft(padded)
    freqs_zp = np.fft.rfftfreq(nfft, d=1 / fs)
    mag_db_zp = 20 * np.log10(np.abs(fft_zp) + 1e-10)
    delta_f = fs / nfft
    axes3[1, 0].plot(
        freqs_zp,
        mag_db_zp,
        linewidth=0.7,
        color=color,
        alpha=0.8,
        label=f"n_fft={nfft} (Δf={delta_f:.1f} Hz)",
    )

axes3[1, 0].set_xlabel("Frequency (Hz)")
axes3[1, 0].set_ylabel("Magnitude (dB)")
axes3[1, 0].set_title("Zero-Padding: Same data, interpolated bins (Hann window)")
axes3[1, 0].set_xlim(0, min(nyquist, 8000))
axes3[1, 0].legend(fontsize=8)
axes3[1, 0].grid(True, alpha=0.3)

# ── Panel D: Visual explanation of the tradeoff ──
axes3[1, 1].axis("off")
tradeoff_text = """
THE FUNDAMENTAL TRADEOFF
━━━━━━━━━━━━━━━━━━━━━━━━

  Frequency resolution:  Δf = fs / N

  To get finer Δf, you need more samples (N).
  More samples = longer time window.

  ┌─────────────────────────────────────────┐
  │  Long window  → Great freq resolution   │
  │                 BUT poor time precision  │
  │                                         │
  │  Short window → Great time precision    │
  │                 BUT poor freq resolution │
  └─────────────────────────────────────────┘

  This is the Heisenberg-Gabor uncertainty
  principle for signals. You CANNOT have
  perfect resolution in both simultaneously.

  This is exactly why the STFT exists —
  it lets you CHOOSE the tradeoff.
"""
axes3[1, 1].text(
    0.05,
    0.95,
    tradeoff_text,
    transform=axes3[1, 1].transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="#f0f9ff", edgecolor="#3b82f6", alpha=0.8),
)

fig3.tight_layout()
fig3.savefig("03_fft_windowing_zeropadding.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 03_fft_windowing_zeropadding.png")


# ======================================================================== #
#                                                                          #
#         PART 4 — THE STFT (Short-Time Fourier Transform)                 #
#                                                                          #
# ======================================================================== #

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WHAT THE STFT DOES                                                      ║
║                                                                          ║
║  The FFT gives you ONE frequency snapshot of the ENTIRE signal.          ║
║  But sound changes over time! A door slam followed by footsteps          ║
║  has different frequencies at different moments.                          ║
║                                                                          ║
║  The STFT solves this by:                                                ║
║    1. Sliding a window across the signal                                 ║
║    2. Computing the FFT of each windowed chunk                           ║
║    3. Stacking the results into a 2D matrix (the spectrogram)            ║
║                                                                          ║
║  ─── ALL THE STFT PARAMETERS ───                                        ║
║                                                                          ║
║  n_fft (also called "window size" or "frame length"):                    ║
║    • Number of samples in each FFT window                                ║
║    • Directly controls the freq/time tradeoff                            ║
║    • Each window produces n_fft/2 + 1 frequency bins                     ║
║    • Frequency resolution: Δf = fs / n_fft                               ║
║    • Time span of one window: n_fft / fs seconds                         ║
║    • Common values: 256, 512, 1024, 2048, 4096                          ║
║                                                                          ║
║  hop_length (also called "hop size" or "step size"):                     ║
║    • How many samples the window moves forward each step                 ║
║    • Smaller hop → more overlap → smoother time axis → more frames       ║
║    • Common choice: n_fft / 4 (75% overlap)                              ║
║    • Number of time frames ≈ (N - n_fft) / hop_length + 1               ║
║    • Time resolution of the spectrogram: hop_length / fs seconds         ║
║                                                                          ║
║  window (window function):                                               ║
║    • Same role as in FFT — reduces spectral leakage                      ║
║    • 'hann' is the standard default for STFT                             ║
║    • Must match n_fft in length                                          ║
║                                                                          ║
║  overlap = n_fft - hop_length:                                           ║
║    • How many samples adjacent windows share                             ║
║    • More overlap → smoother transitions, better reconstruction          ║
║    • 75% overlap (hop = n_fft/4) is standard for analysis                ║
║    • 50% overlap (hop = n_fft/2) is minimum for good reconstruction      ║
║                                                                          ║
║  OUTPUT SHAPE: (n_fft/2 + 1) × num_time_frames                          ║
║    • Rows = frequency bins (0 Hz at top or bottom, Nyquist at other end) ║
║    • Columns = time frames                                               ║
║    • Each cell = complex number (magnitude + phase)                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── STFT with clearly defined parameters ──

n_fft = 1024  # Window size: 1024 samples
hop_length = n_fft // 4  # 256 samples = 75% overlap
window_type = "hann"  # Window function

# Compute STFT using scipy
freqs_stft, times_stft, Zxx = stft(
    signal,
    fs=fs,
    window=window_type,
    nperseg=n_fft,  # = n_fft (window length)
    noverlap=n_fft - hop_length,  # = n_fft - hop_length
    nfft=n_fft,  # FFT size (can be > nperseg for zero-padding)
)

# Convert complex STFT to magnitude spectrogram
magnitude_stft = np.abs(Zxx)
magnitude_stft_db = 20 * np.log10(magnitude_stft + 1e-10)

# Derived parameters
time_resolution = hop_length / fs
freq_resolution_stft = fs / n_fft
overlap = n_fft - hop_length
overlap_pct = overlap / n_fft * 100
window_duration = n_fft / fs
num_freq_bins = magnitude_stft.shape[0]
num_time_frames = magnitude_stft.shape[1]

print(f"\n{'=' * 60}")
print(f"  STFT PARAMETERS")
print(f"{'=' * 60}")
print(f"  n_fft (window size):       {n_fft} samples")
print(f"  Window duration:           {window_duration * 1000:.2f} ms")
print(f"  hop_length:                {hop_length} samples")
print(f"  Overlap:                   {overlap} samples ({overlap_pct:.0f}%)")
print(f"  Window function:           {window_type}")
print(f"  Frequency resolution Δf:   {freq_resolution_stft:.2f} Hz")
print(f"  Time resolution Δt:        {time_resolution * 1000:.2f} ms")
print(f"  Freq bins (rows):          {num_freq_bins}")
print(f"  Time frames (columns):     {num_time_frames}")
print(f"  Spectrogram shape:         {magnitude_stft.shape}")
print(f"  Nyquist frequency:         {nyquist} Hz")
print(f"{'=' * 60}\n")


# ── FIGURE 4: The Spectrogram with annotations ──

fig4, axes4 = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 3]})
fig4.suptitle(
    "FIGURE 4 — STFT Spectrogram: Frequency Over Time", fontsize=14, fontweight="bold"
)

# Top: waveform for reference
axes4[0].plot(t, signal, linewidth=0.3, color="#2563eb")
axes4[0].set_xlabel("Time (s)")
axes4[0].set_ylabel("Amplitude")
axes4[0].set_title("Waveform (for reference — same signal)")
axes4[0].grid(True, alpha=0.3)

# Bottom: spectrogram
# Limit display to useful frequency range
max_display_freq = min(nyquist, 8000)
freq_mask = freqs_stft <= max_display_freq

im = axes4[1].pcolormesh(
    times_stft,
    freqs_stft[freq_mask],
    magnitude_stft_db[freq_mask, :],
    shading="gouraud",
    cmap="magma",
)
axes4[1].set_xlabel("Time (seconds)")
axes4[1].set_ylabel("Frequency (Hz)")
axes4[1].set_title(
    f"Spectrogram  |  n_fft={n_fft}  |  hop={hop_length}  |  "
    f"Δf={freq_resolution_stft:.1f} Hz  |  Δt={time_resolution * 1000:.1f} ms  |  "
    f"window={window_type}  |  overlap={overlap_pct:.0f}%"
)

cbar = fig4.colorbar(im, ax=axes4[1], label="Magnitude (dB)")

fig4.tight_layout()
fig4.savefig("04_stft_spectrogram.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 04_stft_spectrogram.png")


# ======================================================================== #
#                                                                          #
#         PART 5 — THE TIME-FREQUENCY TRADEOFF VISUALIZED                  #
#                                                                          #
# ======================================================================== #

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  THE CORE INSIGHT FOR YOUR PROJECT                                       ║
║                                                                          ║
║  Different n_fft values are better for different sounds:                  ║
║                                                                          ║
║  Small n_fft (256):                                                      ║
║    • Great for transients (door knocks, claps, chopping)                 ║
║    • You can see exactly WHEN the sound happens                          ║
║    • But frequency detail is blurry                                      ║
║                                                                          ║
║  Large n_fft (4096):                                                     ║
║    • Great for steady tones (AC hum, running water, alarms)              ║
║    • You can see exactly WHAT frequencies are present                    ║
║    • But timing is smeared out                                           ║
║                                                                          ║
║  For household sound classification:                                     ║
║    • n_fft = 1024 or 2048 is a good starting point                       ║
║    • YAMNet uses n_fft = 512 at 16 kHz (32 ms windows)                  ║
║    • Your MFCC extraction will use the STFT under the hood               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

nfft_values = [256, 512, 1024, 2048, 4096]

fig5, axes5 = plt.subplots(len(nfft_values), 1, figsize=(14, 3 * len(nfft_values)))
fig5.suptitle(
    "FIGURE 5 — The Time-Frequency Tradeoff: Same Audio, Different n_fft",
    fontsize=14,
    fontweight="bold",
)

for i, nfft_val in enumerate(nfft_values):
    hop_val = nfft_val // 4

    f_s, t_s, Z_s = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nfft_val,
        noverlap=nfft_val - hop_val,
        nfft=nfft_val,
    )

    mag_db_s = 20 * np.log10(np.abs(Z_s) + 1e-10)

    freq_mask_s = f_s <= max_display_freq

    axes5[i].pcolormesh(
        t_s, f_s[freq_mask_s], mag_db_s[freq_mask_s, :], shading="gouraud", cmap="magma"
    )

    delta_f = fs / nfft_val
    delta_t = hop_val / fs
    win_ms = nfft_val / fs * 1000

    axes5[i].set_ylabel("Freq (Hz)")
    axes5[i].set_title(
        f"n_fft = {nfft_val}  |  window = {win_ms:.1f} ms  |  "
        f"Δf = {delta_f:.1f} Hz  |  Δt = {delta_t * 1000:.1f} ms  |  "
        f"freq bins = {sum(freq_mask_s)}  |  time frames = {Z_s.shape[1]}"
    )

axes5[-1].set_xlabel("Time (seconds)")
fig5.tight_layout()
fig5.savefig("05_nfft_tradeoff.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 05_nfft_tradeoff.png")


# ── Show all figures ──
print("\n✓ All 5 figures saved. Opening plots...")
plt.show()

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  QUICK REFERENCE — ALL FOURIER TRANSFORM PARAMETERS                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  SHARED (FFT & STFT):                                                    ║
║    fs .............. Sample rate (Hz)                                     ║
║    N ............... Number of input samples                              ║
║    n_fft ........... FFT size (= N for basic FFT, tunable for STFT)      ║
║    window .......... Window function (hann, hamming, blackman, etc.)      ║
║    Nyquist ......... fs / 2 (max representable frequency)                ║
║    Δf .............. fs / n_fft (frequency resolution)                    ║
║    freq bins ....... n_fft / 2 + 1 (unique positive frequencies)         ║
║                                                                          ║
║  STFT-SPECIFIC:                                                          ║
║    hop_length ...... Samples between successive windows                   ║
║    overlap ......... n_fft - hop_length (shared samples)                  ║
║    Δt .............. hop_length / fs (time resolution)                    ║
║    time frames ..... ≈ (N - n_fft) / hop_length + 1                      ║
║    output shape .... (n_fft/2 + 1) × time_frames                        ║
║                                                                          ║
║  THE TRADEOFF:                                                           ║
║    Large n_fft → fine Δf, coarse Δt (good for steady tones)             ║
║    Small n_fft → coarse Δf, fine Δt (good for transients)               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
