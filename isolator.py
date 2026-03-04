"""
Interactive Audio Feature Isolator
====================================
Three decomposition modes for isolating sound events:

  - Watershed: Instance-level segmentation. Each individual sound event
    (peak in the spectrogram) gets its own mask, like Photoshop's magic
    wand. Best for isolating distinct occurrences.

  - NMF: Source separation via Non-Negative Matrix Factorization.
    Decomposes into overlapping frequency patterns. Best for separating
    sound types that overlap in time.

  - CCA: Connected Component Analysis. Simple threshold-based blob
    detection. Fast but can't handle overlapping events.

Authors: Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
Course:  TECHIN 513A — Managing Data And Signal Processing
"""

import tempfile
import subprocess
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.ndimage import (label, binary_closing, binary_opening,
                           gaussian_filter, binary_dilation, binary_erosion)
import sounddevice as sd


# ── Audio loading ──────────────────────────────────────────────────────────

def load_audio(path):
    """Load an audio file and return (signal, sample_rate) as float64 mono."""
    try:
        fs, raw = wavfile.read(path)
    except ValueError:
        out_path = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, out_path],
            capture_output=True, check=True,
        )
        fs, raw = wavfile.read(out_path)

    if raw.ndim == 2:
        raw = raw.mean(axis=1)

    if raw.dtype == np.int16:
        sig = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        sig = raw.astype(np.float64) / 2147483648.0
    elif raw.dtype in (np.float32, np.float64):
        sig = raw.astype(np.float64)
    else:
        sig = raw.astype(np.float64) / max(np.max(np.abs(raw)), 1)

    return sig, fs


# ── NMF (pure numpy) ──────────────────────────────────────────────────────

def nmf(V, n_components, max_iter=150, tol=1e-4):
    """
    Non-Negative Matrix Factorization using multiplicative update rules.

    Decomposes V ≈ W @ H where:
      - V: (n_freqs, n_times) magnitude spectrogram (non-negative)
      - W: (n_freqs, n_components) frequency basis vectors
      - H: (n_components, n_times) time activation vectors
      - Component k's spectrogram = W[:, k:k+1] @ H[k:k+1, :]

    Each component represents an additive "source" that can overlap with others.
    """
    np.random.seed(42)
    n_f, n_t = V.shape
    W = np.random.rand(n_f, n_components) + 0.1
    H = np.random.rand(n_components, n_t) + 0.1

    eps = 1e-10
    prev_cost = float("inf")

    for iteration in range(max_iter):
        # Update H
        WtV = W.T @ V
        WtWH = W.T @ W @ H + eps
        H *= WtV / WtWH

        # Update W
        VHt = V @ H.T
        WHHt = W @ H @ H.T + eps
        W *= VHt / WHHt

        # Check convergence every 10 iterations
        if iteration % 10 == 0:
            cost = np.sum((V - W @ H) ** 2)
            if abs(prev_cost - cost) / (prev_cost + eps) < tol:
                break
            prev_cost = cost

    return W, H


# ── Watershed segmentation (pure scipy) ───────────────────────────────────

def watershed_segment(mag_db, threshold_db, peak_dist=10, min_size=30):
    """
    Watershed segmentation of a spectrogram.

    1. Threshold to remove background noise
    2. Find local peaks (energy maxima) — these become "seeds"
    3. Flood-fill from each peak downhill to create basins
    4. Each basin = one distinct sound event instance

    Returns labeled array where each pixel has an integer label (0 = background).
    """
    from scipy.ndimage import (maximum_filter, label as nd_label,
                               find_objects, sum as nd_sum)

    # Binary foreground
    foreground = mag_db >= threshold_db

    # Find local maxima (peaks) — these are the "seeds" for watershed
    # A peak is a point that equals the local maximum in its neighborhood
    neighborhood = np.ones((peak_dist, peak_dist))
    local_max = maximum_filter(mag_db, footprint=neighborhood)
    peaks = (mag_db == local_max) & foreground

    # Label the peaks as markers
    markers, n_markers = nd_label(peaks)

    # Simple watershed: iteratively expand markers into foreground
    # by processing pixels from highest to lowest energy
    labeled = markers.copy()

    # Get all foreground pixel coordinates sorted by descending energy
    fg_coords = np.argwhere(foreground)
    if len(fg_coords) == 0:
        return np.zeros_like(mag_db, dtype=int), 0

    fg_energies = mag_db[fg_coords[:, 0], fg_coords[:, 1]]
    sort_idx = np.argsort(-fg_energies)  # highest energy first
    fg_coords = fg_coords[sort_idx]

    # Expand: for each pixel (high-to-low energy), assign it the label
    # of its nearest already-labeled neighbor
    from scipy.ndimage import maximum_filter as mf
    # Iterative expansion (fast — typically converges in <10 passes)
    for _ in range(max(mag_db.shape)):
        # Expand labeled regions by 1 pixel in all directions
        expanded = maximum_filter(labeled, size=3)
        # Only fill unlabeled foreground pixels
        fill_mask = (labeled == 0) & foreground & (expanded > 0)
        if not np.any(fill_mask):
            break
        labeled[fill_mask] = expanded[fill_mask]

    # Filter out tiny regions
    n_labels = labeled.max()
    for lbl in range(1, n_labels + 1):
        if np.sum(labeled == lbl) < min_size:
            labeled[labeled == lbl] = 0

    # Re-label contiguously
    unique_labels = np.unique(labeled[labeled > 0])
    relabeled = np.zeros_like(labeled)
    for new_id, old_id in enumerate(unique_labels, 1):
        relabeled[labeled == old_id] = new_id

    return relabeled, int(relabeled.max())


# ── Color palette ──────────────────────────────────────────────────────────

COLORS = [
    "#22d3ee", "#f97316", "#a78bfa", "#34d399", "#f43f5e",
    "#facc15", "#38bdf8", "#fb923c", "#c084fc", "#4ade80",
    "#e879f9", "#2dd4bf",
]


# ── Isolator ───────────────────────────────────────────────────────────────

class Isolator:
    def __init__(self, signal, fs, n_fft=1024, overlap_pct=75, window="hann",
                 max_freq=8000):
        self.signal = signal
        self.fs = fs
        self.n_fft = n_fft
        self.overlap_pct = overlap_pct
        self.window = window
        self.max_freq = min(max_freq, fs / 2)
        self.duration = len(signal) / fs

        # STFT
        hop = max(1, int(n_fft * (1 - overlap_pct / 100)))
        self.hop = hop
        self.freqs, self.times, self.Zxx = stft(
            signal, fs=fs, window=window,
            nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
        )
        self.mag = np.abs(self.Zxx)
        self.mag_db = 20 * np.log10(self.mag + 1e-10)
        self.phase = np.angle(self.Zxx)

        # Display mask (freq range)
        self.disp_mask = self.freqs <= self.max_freq
        self.n_disp_freqs = np.sum(self.disp_mask)

        # State
        self.components = []     # list of dicts
        self.selected_idx = None
        self.isolated = None
        self.mode = "watershed"  # "watershed", "nmf", or "cca"
        self.overlay_img = None  # RGBA overlay artist

        self._build_gui()
        self._decompose()

    # ── GUI ──

    def _build_gui(self):
        self.fig = plt.figure(figsize=(17, 10), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Audio Feature Isolator")

        # Main panels
        gs = self.fig.add_gridspec(
            3, 1, height_ratios=[1, 3, 1],
            left=0.06, right=0.65, top=0.93, bottom=0.32, hspace=0.3,
        )
        self.ax_wave = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1])
        self.ax_iso = self.fig.add_subplot(gs[2])

        for ax in (self.ax_wave, self.ax_spec, self.ax_iso):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#aaa", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#333")

        # Waveform
        t = np.arange(len(self.signal)) / self.fs
        self.ax_wave.plot(t, self.signal, lw=0.3, color="#2563eb")
        self.ax_wave.set_ylabel("Amp", color="#aaa", fontsize=8)
        self.ax_wave.set_title("Original Waveform", color="#ccc", fontsize=9)
        self.ax_wave.set_xlim(0, self.duration)

        # Spectrogram (imshow for speed)
        disp_db = self.mag_db[self.disp_mask, :]
        self.spec_img = self.ax_spec.imshow(
            disp_db, aspect="auto", origin="lower", cmap="magma",
            extent=[0, self.duration, 0, self.max_freq],
            interpolation="bilinear",
        )
        self.ax_spec.set_ylabel("Freq (Hz)", color="#aaa", fontsize=8)
        self.ax_spec.set_title("Click a component to isolate", color="#ccc", fontsize=9)

        # RGBA overlay for colored components (fast — single image)
        blank = np.zeros((self.n_disp_freqs, len(self.times), 4))
        self.overlay_img = self.ax_spec.imshow(
            blank, aspect="auto", origin="lower",
            extent=[0, self.duration, 0, self.max_freq],
            interpolation="nearest",
        )

        # Isolated waveform
        self.iso_line, = self.ax_iso.plot([], [], lw=0.4, color="#22d3ee")
        self.ax_iso.set_ylabel("Amp", color="#aaa", fontsize=8)
        self.ax_iso.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        self.ax_iso.set_title("Click a component above", color="#666", fontsize=9)
        self.ax_iso.set_xlim(0, self.duration)

        # ── Info panel ──
        self.ax_info = self.fig.add_axes([0.68, 0.32, 0.30, 0.61], facecolor="#16213e")
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for sp in self.ax_info.spines.values():
            sp.set_color("#333")
        self.info_text = self.ax_info.text(
            0.04, 0.97, "", transform=self.ax_info.transAxes,
            fontsize=7.5, color="#ccc", va="top", fontfamily="monospace",
        )

        # ── Mode selector ──
        sc = "#0f3460"
        ax_mode = self.fig.add_axes([0.68, 0.20, 0.10, 0.11], facecolor="#16213e")
        self.radio_mode = RadioButtons(
            ax_mode, ["Watershed", "NMF", "CCA"], active=0)
        for lbl in self.radio_mode.labels:
            lbl.set_color("#ccc")
            lbl.set_fontsize(8)
        self.radio_mode.on_clicked(self._on_mode_change)

        # ── Sliders ──
        # Row positions for mode-specific sliders (stacked, toggled)
        r1, r2, r3 = 0.22, 0.19, 0.16

        # Watershed sliders
        ax_wt = self.fig.add_axes([0.06, r1, 0.55, 0.022], facecolor=sc)
        ax_wp = self.fig.add_axes([0.06, r2, 0.55, 0.022], facecolor=sc)
        ax_wm = self.fig.add_axes([0.06, r3, 0.55, 0.022], facecolor=sc)
        # NMF sliders
        ax_nc = self.fig.add_axes([0.06, r1, 0.55, 0.022], facecolor=sc)
        ax_it = self.fig.add_axes([0.06, r2, 0.55, 0.022], facecolor=sc)
        # CCA sliders
        ax_th = self.fig.add_axes([0.06, r1, 0.55, 0.022], facecolor=sc)
        ax_mp = self.fig.add_axes([0.06, r2, 0.55, 0.022], facecolor=sc)
        ax_ms = self.fig.add_axes([0.06, r3, 0.55, 0.022], facecolor=sc)
        # Shared sliders
        ax_sf = self.fig.add_axes([0.06, 0.13, 0.55, 0.022], facecolor=sc)
        ax_gr = self.fig.add_axes([0.06, 0.10, 0.55, 0.022], facecolor=sc)
        ax_vol = self.fig.add_axes([0.06, 0.07, 0.55, 0.022], facecolor=sc)

        db_min = float(np.floor(self.mag_db.min()))
        db_max = float(np.ceil(self.mag_db.max()))
        default_thresh = float(np.percentile(self.mag_db, 40))

        # Watershed
        self.s_ws_thresh = Slider(ax_wt, "Threshold (dB)", db_min, db_max,
                                  valinit=default_thresh, color="#ec4899", valstep=1)
        self.s_ws_peak_dist = Slider(ax_wp, "Peak Distance", 3, 50, valinit=10,
                                     color="#f97316", valstep=1)
        self.s_ws_min_size = Slider(ax_wm, "Min Size", 5, 500, valinit=30,
                                    color="#a78bfa", valstep=5)
        # NMF
        self.s_ncomp = Slider(ax_nc, "Components", 2, 15, valinit=5,
                              color="#ec4899", valstep=1)
        self.s_nmf_iter = Slider(ax_it, "NMF Iterations", 20, 300, valinit=100,
                                 color="#f97316", valstep=10)
        # CCA
        self.s_thresh = Slider(ax_th, "Threshold (dB)", db_min, db_max,
                               valinit=default_thresh, color="#ec4899", valstep=1)
        self.s_morph = Slider(ax_mp, "Morph Close", 1, 15, valinit=3,
                              color="#f97316", valstep=1)
        self.s_min_size = Slider(ax_ms, "Min Size", 5, 500, valinit=50,
                                 color="#a78bfa", valstep=5)
        # Shared
        self.s_soft = Slider(ax_sf, "Edge Softness", 0, 5, valinit=1.0,
                             color="#34d399", valstep=0.25)
        self.s_grow = Slider(ax_gr, "Grow/Shrink", -10, 10, valinit=0,
                             color="#facc15", valstep=1)
        self.s_vol = Slider(ax_vol, "Volume", 0, 1.0, valinit=0.5,
                            color="#38bdf8", valstep=0.05)

        self.ws_sliders = [self.s_ws_thresh, self.s_ws_peak_dist, self.s_ws_min_size]
        self.nmf_sliders = [self.s_ncomp, self.s_nmf_iter]
        self.cca_sliders = [self.s_thresh, self.s_morph, self.s_min_size]
        self.all_sliders = (self.ws_sliders + self.nmf_sliders +
                            self.cca_sliders +
                            [self.s_soft, self.s_grow, self.s_vol])

        for s in self.all_sliders:
            s.label.set_color("#ccc")
            s.label.set_fontsize(8)
            s.valtext.set_color("#ccc")

        for s in self.ws_sliders:
            s.on_changed(self._on_decompose_change)
        self.s_ncomp.on_changed(self._on_decompose_change)
        self.s_nmf_iter.on_changed(self._on_decompose_change)
        self.s_thresh.on_changed(self._on_decompose_change)
        self.s_morph.on_changed(self._on_decompose_change)
        self.s_min_size.on_changed(self._on_decompose_change)
        self.s_soft.on_changed(self._on_soft_change)
        self.s_grow.on_changed(self._on_grow_change)

        self._toggle_slider_visibility()

        # ── Buttons ──
        ax_po = self.fig.add_axes([0.80, 0.24, 0.09, 0.035], facecolor=sc)
        ax_pi = self.fig.add_axes([0.90, 0.24, 0.09, 0.035], facecolor=sc)
        ax_st = self.fig.add_axes([0.80, 0.20, 0.09, 0.035], facecolor=sc)
        ax_ex = self.fig.add_axes([0.90, 0.20, 0.09, 0.035], facecolor=sc)
        ax_ea = self.fig.add_axes([0.80, 0.16, 0.19, 0.035], facecolor=sc)

        self.b_play_orig = Button(ax_po, "Play Orig", color=sc, hovercolor="#1a1a5e")
        self.b_play_sel = Button(ax_pi, "Play Sel", color=sc, hovercolor="#1a1a5e")
        self.b_stop = Button(ax_st, "Stop", color=sc, hovercolor="#1a1a5e")
        self.b_export = Button(ax_ex, "Export Sel", color=sc, hovercolor="#1a1a5e")
        self.b_export_all = Button(ax_ea, "Export All", color=sc, hovercolor="#1a1a5e")

        for b in (self.b_play_orig, self.b_play_sel, self.b_stop,
                  self.b_export, self.b_export_all):
            b.label.set_color("#ccc")
            b.label.set_fontsize(8)

        self.b_play_orig.on_clicked(self.play_original)
        self.b_play_sel.on_clicked(self.play_selected)
        self.b_stop.on_clicked(self.stop_playback)
        self.b_export.on_clicked(self.export_selected)
        self.b_export_all.on_clicked(self.export_all)

        self.legend = None
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.fig.suptitle(
            f"Audio Feature Isolator  |  {self.fs} Hz  |  "
            f"{self.duration:.2f}s  |  n_fft={self.n_fft}",
            color="#e2e8f0", fontsize=11, fontweight="bold",
        )

    def _toggle_slider_visibility(self):
        for s in self.ws_sliders:
            s.ax.set_visible(self.mode == "watershed")
        for s in self.nmf_sliders:
            s.ax.set_visible(self.mode == "nmf")
        for s in self.cca_sliders:
            s.ax.set_visible(self.mode == "cca")
        self.fig.canvas.draw_idle()

    # ── Decomposition ──

    def _on_mode_change(self, label):
        self.mode = label.lower()
        self._toggle_slider_visibility()
        self.s_grow.set_val(0)
        self._decompose()

    def _on_decompose_change(self, _val):
        self.s_grow.set_val(0)
        self._decompose()

    def _on_soft_change(self, _val):
        # Re-apply softness without full redecompose
        if self.components:
            self._rebuild_audio_from_masks()
            self._draw_overlay()

    def _on_grow_change(self, _val):
        if self.selected_idx is None:
            return
        grow = int(self.s_grow.val)
        comp = self.components[self.selected_idx]
        mask = comp["base_mask"].copy()

        if grow > 0:
            struct = np.ones((3, 3))
            for _ in range(grow):
                mask = binary_dilation(mask, structure=struct).astype(float)
        elif grow < 0:
            struct = np.ones((3, 3))
            for _ in range(abs(grow)):
                mask = binary_erosion(mask, structure=struct).astype(float)

        comp["mask"] = mask
        self._rebuild_single_audio(self.selected_idx)
        self._draw_overlay()
        self._select_component(self.selected_idx)

    def _decompose(self):
        """Run NMF or CCA decomposition."""
        print(f"Decomposing ({self.mode})...", end=" ", flush=True)

        if self.mode == "watershed":
            self._decompose_watershed()
        elif self.mode == "nmf":
            self._decompose_nmf()
        else:
            self._decompose_cca()

        self._rebuild_audio_from_masks()
        self.selected_idx = None
        self.isolated = None
        self.iso_line.set_data([], [])
        self.ax_iso.set_title("Click a component above", color="#666", fontsize=9)

        self._draw_overlay()
        self._update_info()
        self.fig.canvas.draw_idle()
        print(f"{len(self.components)} components")

    def _decompose_watershed(self):
        threshold = self.s_ws_thresh.val
        peak_dist = int(self.s_ws_peak_dist.val)
        min_size = int(self.s_ws_min_size.val)

        labeled, n_found = watershed_segment(
            self.mag_db, threshold, peak_dist=peak_dist, min_size=min_size)

        self.components = []
        for comp_id in range(1, n_found + 1):
            comp_mask = (labeled == comp_id).astype(np.float64)
            if np.sum(comp_mask) == 0:
                continue

            fi, ti = np.where(comp_mask > 0.5)
            color = COLORS[len(self.components) % len(COLORS)]

            # Find peak frequency (where energy is highest in this region)
            region_energy = self.mag_db * comp_mask
            region_energy[comp_mask < 0.5] = -200
            peak_idx = np.unravel_index(np.argmax(region_energy), region_energy.shape)

            self.components.append({
                "mask": comp_mask,
                "base_mask": comp_mask.copy(),
                "color": color,
                "freq_peak": float(self.freqs[peak_idx[0]]),
                "energy": float(np.sum(self.mag[comp_mask > 0.5])),
                "time_min": float(self.times[ti.min()]),
                "time_max": float(self.times[ti.max()]),
                "freq_min": float(self.freqs[fi.min()]),
                "freq_max": float(self.freqs[fi.max()]),
            })

        # Sort by time
        self.components.sort(key=lambda c: c["time_min"])
        for i, c in enumerate(self.components):
            c["color"] = COLORS[i % len(COLORS)]

    def _decompose_nmf(self):
        n_comp = int(self.s_ncomp.val)
        n_iter = int(self.s_nmf_iter.val)

        W, H = nmf(self.mag, n_comp, max_iter=n_iter)

        # Build masks from NMF: each component's proportion of the total
        WH = W @ H + 1e-10
        self.components = []
        for k in range(n_comp):
            comp_mag = np.outer(W[:, k], H[k, :])
            # Soft mask: this component's proportion of the reconstruction
            mask = comp_mag / WH
            color = COLORS[k % len(COLORS)]

            # Find dominant frequency and time ranges
            energy = np.sum(comp_mag, axis=1)
            time_energy = np.sum(comp_mag, axis=0)
            freq_peak = self.freqs[np.argmax(energy)]
            # Time range where this component is most active
            active = time_energy > np.max(time_energy) * 0.1
            t_indices = np.where(active)[0]

            self.components.append({
                "mask": mask,
                "base_mask": mask.copy(),
                "color": color,
                "freq_peak": freq_peak,
                "energy": float(np.sum(comp_mag)),
                "time_min": float(self.times[t_indices[0]]) if len(t_indices) > 0 else 0,
                "time_max": float(self.times[t_indices[-1]]) if len(t_indices) > 0 else self.duration,
                "W_col": W[:, k],
                "H_row": H[k, :],
            })

        # Sort by energy (strongest first)
        self.components.sort(key=lambda c: -c["energy"])
        for i, c in enumerate(self.components):
            c["color"] = COLORS[i % len(COLORS)]

    def _decompose_cca(self):
        threshold = self.s_thresh.val
        morph_size = int(self.s_morph.val)
        min_size = int(self.s_min_size.val)

        binary = self.mag_db >= threshold
        struct = np.ones((morph_size, morph_size))
        binary = binary_closing(binary, structure=struct)
        binary = binary_opening(binary, structure=np.ones((2, 2)))

        labeled, n_found = label(binary)

        self.components = []
        for comp_id in range(1, n_found + 1):
            comp_mask = (labeled == comp_id).astype(np.float64)
            if np.sum(comp_mask) < min_size:
                continue

            fi, ti = np.where(comp_mask > 0.5)
            color = COLORS[len(self.components) % len(COLORS)]

            self.components.append({
                "mask": comp_mask,
                "base_mask": comp_mask.copy(),
                "color": color,
                "freq_peak": float(self.freqs[fi[np.argmax(
                    self.mag_db[fi, ti])]]),
                "energy": float(np.sum(self.mag[comp_mask > 0.5])),
                "time_min": float(self.times[ti.min()]),
                "time_max": float(self.times[ti.max()]),
            })

        self.components.sort(key=lambda c: c["time_min"])
        for i, c in enumerate(self.components):
            c["color"] = COLORS[i % len(COLORS)]

    def _rebuild_audio_from_masks(self):
        softness = self.s_soft.val
        for comp in self.components:
            self._rebuild_single_audio_from(comp, softness)

    def _rebuild_single_audio(self, idx):
        self._rebuild_single_audio_from(self.components[idx], self.s_soft.val)

    def _rebuild_single_audio_from(self, comp, softness):
        mask = comp["mask"]
        if softness > 0:
            soft = gaussian_filter(mask, sigma=softness)
        else:
            soft = mask

        # Reconstruct: apply mask to magnitude, keep original phase
        masked_stft = soft * self.mag * np.exp(1j * self.phase)
        _, audio = istft(
            masked_stft, fs=self.fs, window=self.window,
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop,
            nfft=self.n_fft,
        )
        comp["audio"] = audio[:len(self.signal)]

    # ── Fast overlay rendering ──

    def _draw_overlay(self):
        """Draw all components as a single RGBA image overlay (fast)."""
        h = self.n_disp_freqs
        w = len(self.times)
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        for i, comp in enumerate(self.components):
            r, g, b, _ = to_rgba(comp["color"])
            mask_disp = comp["mask"][:h, :].astype(np.float32)

            # Normalize mask to [0, 1] range for display
            m_max = mask_disp.max()
            if m_max > 0:
                mask_norm = mask_disp / m_max
            else:
                mask_norm = mask_disp

            alpha = mask_norm * 0.35
            # Additive blend
            rgba[:, :, 0] += r * alpha
            rgba[:, :, 1] += g * alpha
            rgba[:, :, 2] += b * alpha
            rgba[:, :, 3] = np.clip(rgba[:, :, 3] + alpha, 0, 0.7)

        # Highlight selected component border
        if self.selected_idx is not None and self.selected_idx < len(self.components):
            comp = self.components[self.selected_idx]
            mask_disp = comp["mask"][:h, :].astype(np.float32)
            m_max = mask_disp.max()
            if m_max > 0:
                # Edge detection via gradient
                from scipy.ndimage import sobel
                edge = np.sqrt(
                    sobel(mask_disp / m_max, axis=0)**2 +
                    sobel(mask_disp / m_max, axis=1)**2
                )
                edge = (edge > 0.1).astype(np.float32)
                r, g, b, _ = to_rgba(comp["color"])
                rgba[:, :, 0] = np.where(edge > 0, r, rgba[:, :, 0])
                rgba[:, :, 1] = np.where(edge > 0, g, rgba[:, :, 1])
                rgba[:, :, 2] = np.where(edge > 0, b, rgba[:, :, 2])
                rgba[:, :, 3] = np.where(edge > 0, 1.0, rgba[:, :, 3])

        rgba = np.clip(rgba, 0, 1)
        self.overlay_img.set_data(rgba)

        # Legend
        if self.legend:
            self.legend.remove()
        if self.components:
            patches = []
            for i, c in enumerate(self.components):
                sel = " <<" if i == self.selected_idx else ""
                lbl = f"{i+1}: {c['freq_peak']:.0f}Hz peak{sel}"
                patches.append(Patch(facecolor=c["color"], alpha=0.6, label=lbl))
            self.legend = self.ax_spec.legend(
                handles=patches, loc="upper right", fontsize=6.5,
                facecolor="#1a1a2ecc", edgecolor="#333", labelcolor="#ccc",
            )

        title = f"{len(self.components)} components ({self.mode.upper()}) — click to isolate"
        self.ax_spec.set_title(title, color="#ccc", fontsize=9)
        self.fig.canvas.draw_idle()

    # ── Info panel ──

    def _update_info(self):
        lines = [
            f"MODE: {self.mode.upper()}",
            "=" * 30,
        ]

        if self.mode == "watershed":
            lines += [
                f"Threshold:    {self.s_ws_thresh.val:>6.0f} dB",
                f"Peak Dist:    {int(self.s_ws_peak_dist.val):>6d} px",
                f"Min Size:     {int(self.s_ws_min_size.val):>6d} px",
            ]
        elif self.mode == "nmf":
            lines += [
                f"Components:   {int(self.s_ncomp.val):>6d}",
                f"Iterations:   {int(self.s_nmf_iter.val):>6d}",
            ]
        else:
            lines += [
                f"Threshold:    {self.s_thresh.val:>6.0f} dB",
                f"Morph Close:  {int(self.s_morph.val):>6d} px",
                f"Min Size:     {int(self.s_min_size.val):>6d} px",
            ]

        lines += [
            f"Edge Softness:{self.s_soft.val:>6.2f} σ",
            f"Volume:       {self.s_vol.val:>6.2f}",
            "",
            "STFT",
            "-" * 30,
            f"n_fft:        {self.n_fft:>6d}",
            f"Hop:          {self.hop:>6d}",
            f"Overlap:      {self.overlap_pct:>5.0f}%",
            f"Window:       {self.window:>6s}",
            f"Sample rate:  {self.fs:>6d} Hz",
            f"Δf:           {self.fs/self.n_fft:>6.1f} Hz",
            f"Δt:           {self.hop/self.fs*1000:>5.1f} ms",
            "",
            f"COMPONENTS: {len(self.components)}",
            "-" * 30,
        ]

        if self.selected_idx is not None and self.selected_idx < len(self.components):
            c = self.components[self.selected_idx]
            grow = self.s_grow.val
            lines += [
                f">> SELECTED: #{self.selected_idx+1} <<",
                f"Peak freq:  {c['freq_peak']:.0f} Hz",
                f"Time range: {c['time_min']:.2f}-{c['time_max']:.2f}s",
                f"Energy:     {c['energy']:.1f}",
                f"Grow/Shrink:{grow:+.0f}",
            ]
        else:
            lines.append("Click a component to select")

        if self.components:
            lines += ["", "ALL COMPONENTS:"]
            for i, c in enumerate(self.components):
                sel = ">>" if i == self.selected_idx else "  "
                lines.append(
                    f"{sel}{i+1}: {c['freq_peak']:>5.0f}Hz  "
                    f"{c['time_min']:.1f}-{c['time_max']:.1f}s"
                )

        self.info_text.set_text("\n".join(lines))

    # ── Click to select ──

    def _on_click(self, event):
        if event.inaxes != self.ax_spec:
            return
        if event.xdata is None or event.ydata is None:
            return

        t_idx = np.argmin(np.abs(self.times - event.xdata))
        f_idx = np.argmin(np.abs(self.freqs - event.ydata))

        # Find component with highest mask value at click point
        best_idx = None
        best_val = 0
        for i, comp in enumerate(self.components):
            val = comp["mask"][f_idx, t_idx]
            if val > best_val:
                best_val = val
                best_idx = i

        if best_idx is not None and best_val > 0.01:
            self.s_grow.set_val(0)
            self._select_component(best_idx)
        else:
            self.selected_idx = None
            self.isolated = None
            self.iso_line.set_data([], [])
            self.ax_iso.set_title("No component here", color="#666", fontsize=9)
            self._draw_overlay()
            self._update_info()
            self.fig.canvas.draw_idle()

    def _select_component(self, idx):
        self.selected_idx = idx
        comp = self.components[idx]
        self.isolated = comp["audio"]

        t = np.arange(len(self.isolated)) / self.fs
        self.iso_line.set_data(t, self.isolated)
        self.iso_line.set_color(comp["color"])
        peak = max(np.max(np.abs(self.isolated)) * 1.1, 0.01)
        self.ax_iso.set_ylim(-peak, peak)
        self.ax_iso.set_title(
            f"Component {idx+1}: {comp['freq_peak']:.0f} Hz peak, "
            f"{comp['time_min']:.1f}-{comp['time_max']:.1f}s",
            color=comp["color"], fontsize=9,
        )

        self._draw_overlay()
        self._update_info()
        self.fig.canvas.draw_idle()
        print(f"Selected component {idx+1}")

    # ── Playback (volume-normalized) ──

    def _play(self, audio):
        sd.stop()
        vol = self.s_vol.val
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return
        normalized = (audio / peak * vol).astype(np.float32)
        sd.play(normalized, self.fs)

    def play_original(self, _event):
        self._play(self.signal)

    def play_selected(self, _event):
        if self.isolated is None:
            print("No component selected")
            return
        self._play(self.isolated)

    def stop_playback(self, _event):
        sd.stop()

    # ── Export ──

    def _export_audio(self, audio, name):
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            print(f"Skipping {name} — silent")
            return
        clip_int = np.int16(audio / peak * 32767 * 0.9)
        wavfile.write(name, self.fs, clip_int)
        print(f"Exported: {name}")

    def export_selected(self, _event):
        if self.selected_idx is None:
            print("No component selected")
            return
        c = self.components[self.selected_idx]
        name = f"component_{self.selected_idx+1}_{c['freq_peak']:.0f}Hz.wav"
        self._export_audio(c["audio"], name)

    def export_all(self, _event):
        for i, c in enumerate(self.components):
            name = f"component_{i+1}_{c['freq_peak']:.0f}Hz.wav"
            self._export_audio(c["audio"], name)
        print(f"Exported {len(self.components)} components")

    def show(self):
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audio Feature Isolator")
    parser.add_argument("audio_file", help="Audio file path")
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=75)
    parser.add_argument("--window", type=str, default="hann")
    parser.add_argument("--maxfreq", type=float, default=8000)
    args = parser.parse_args()

    signal, fs = load_audio(args.audio_file)
    print(f"Loaded: {args.audio_file}")
    print(f"  {fs} Hz | {len(signal)/fs:.2f}s | {len(signal)} samples")

    Isolator(signal, fs, n_fft=args.nfft, overlap_pct=args.overlap,
             window=args.window, max_freq=args.maxfreq).show()


if __name__ == "__main__":
    main()
