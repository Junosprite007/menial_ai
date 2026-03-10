"""Audio decomposition service — extracts algorithms from isolator.py for headless use."""

import os
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.ndimage import (
    label, binary_closing, binary_opening, gaussian_filter,
    maximum_filter, find_objects, sum as nd_sum,
)

from app.config import JOBS_DIR
from app.utils.audio import load_audio

COLORS = [
    "#22d3ee", "#f97316", "#a78bfa", "#34d399", "#f43f5e",
    "#facc15", "#38bdf8", "#fb923c", "#c084fc", "#4ade80",
    "#e879f9", "#2dd4bf",
]


def nmf(V, n_components, max_iter=150, tol=1e-4):
    """NMF using multiplicative update rules. V approx W @ H."""
    np.random.seed(42)
    n_f, n_t = V.shape
    W = np.random.rand(n_f, n_components) + 0.1
    H = np.random.rand(n_components, n_t) + 0.1
    eps = 1e-10
    prev_cost = float("inf")

    for iteration in range(max_iter):
        WtV = W.T @ V
        WtWH = W.T @ W @ H + eps
        H *= WtV / WtWH

        VHt = V @ H.T
        WHHt = W @ H @ H.T + eps
        W *= VHt / WHHt

        if iteration % 10 == 0:
            cost = np.sum((V - W @ H) ** 2)
            if abs(prev_cost - cost) / (prev_cost + eps) < tol:
                break
            prev_cost = cost

    return W, H


def watershed_segment(mag_db, threshold_db, peak_dist=10, min_size=30):
    """Watershed segmentation of a spectrogram."""
    foreground = mag_db >= threshold_db
    neighborhood = np.ones((peak_dist, peak_dist))
    local_max = maximum_filter(mag_db, footprint=neighborhood)
    peaks = (mag_db == local_max) & foreground
    markers, n_markers = label(peaks)
    labeled = markers.copy()

    fg_coords = np.argwhere(foreground)
    if len(fg_coords) == 0:
        return np.zeros_like(mag_db, dtype=int), 0

    for _ in range(max(mag_db.shape)):
        expanded = maximum_filter(labeled, size=3)
        fill_mask = (labeled == 0) & foreground & (expanded > 0)
        if not np.any(fill_mask):
            break
        labeled[fill_mask] = expanded[fill_mask]

    n_labels = labeled.max()
    for lbl in range(1, n_labels + 1):
        if np.sum(labeled == lbl) < min_size:
            labeled[labeled == lbl] = 0

    unique_labels = np.unique(labeled[labeled > 0])
    relabeled = np.zeros_like(labeled)
    for new_id, old_id in enumerate(unique_labels, 1):
        relabeled[labeled == old_id] = new_id

    return relabeled, int(relabeled.max())


def _decompose_nmf(mag, freqs, times, params):
    """NMF decomposition returning component list."""
    n_comp = params["n_components"]
    n_iter = params["nmf_iter"]
    W, H = nmf(mag, n_comp, max_iter=n_iter)
    WH = W @ H + 1e-10
    components = []

    for k in range(n_comp):
        comp_mag = np.outer(W[:, k], H[k, :])
        mask = comp_mag / WH
        energy = np.sum(comp_mag, axis=1)
        freq_peak = freqs[np.argmax(energy)]
        time_energy = np.sum(comp_mag, axis=0)
        active = time_energy > np.max(time_energy) * 0.1
        t_idx = np.where(active)[0]

        components.append({
            "mask": mask,
            "color": COLORS[k % len(COLORS)],
            "freq_peak": float(freq_peak),
            "energy": float(np.sum(comp_mag)),
            "time_min": float(times[t_idx[0]]) if len(t_idx) > 0 else 0,
            "time_max": float(times[t_idx[-1]]) if len(t_idx) > 0 else float(times[-1]),
        })

    components.sort(key=lambda c: -c["energy"])
    for i, c in enumerate(components):
        c["color"] = COLORS[i % len(COLORS)]
    return components


def _decompose_watershed(mag, mag_db, freqs, times, params):
    """Watershed decomposition returning component list."""
    labeled, n_found = watershed_segment(
        mag_db, params["threshold"],
        peak_dist=params["peak_dist"],
        min_size=params["min_size"],
    )
    components = []
    for comp_id in range(1, n_found + 1):
        comp_mask = (labeled == comp_id).astype(np.float64)
        if np.sum(comp_mask) == 0:
            continue
        fi, ti = np.where(comp_mask > 0.5)
        region_energy = mag_db * comp_mask
        region_energy[comp_mask < 0.5] = -200
        peak_idx = np.unravel_index(np.argmax(region_energy), region_energy.shape)

        components.append({
            "mask": comp_mask,
            "color": COLORS[len(components) % len(COLORS)],
            "freq_peak": float(freqs[peak_idx[0]]),
            "energy": float(np.sum(mag[comp_mask > 0.5])),
            "time_min": float(times[ti.min()]),
            "time_max": float(times[ti.max()]),
        })

    components.sort(key=lambda c: c["time_min"])
    for i, c in enumerate(components):
        c["color"] = COLORS[i % len(COLORS)]
    return components


def _decompose_cca(mag, mag_db, freqs, times, params):
    """Connected Component Analysis decomposition."""
    threshold = params["threshold"]
    morph_size = params["morph_size"]
    min_size = params["min_size"]

    binary = mag_db >= threshold
    struct = np.ones((morph_size, morph_size))
    binary = binary_closing(binary, structure=struct)
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    labeled, n_found = label(binary)

    components = []
    for comp_id in range(1, n_found + 1):
        comp_mask = (labeled == comp_id).astype(np.float64)
        if np.sum(comp_mask) < min_size:
            continue
        fi, ti = np.where(comp_mask > 0.5)
        components.append({
            "mask": comp_mask,
            "color": COLORS[len(components) % len(COLORS)],
            "freq_peak": float(freqs[fi[np.argmax(mag_db[fi, ti])]]),
            "energy": float(np.sum(mag[comp_mask > 0.5])),
            "time_min": float(times[ti.min()]),
            "time_max": float(times[ti.max()]),
        })

    components.sort(key=lambda c: c["time_min"])
    for i, c in enumerate(components):
        c["color"] = COLORS[i % len(COLORS)]
    return components


def _render_spectrogram(mag_db, freqs, times, max_freq, components, output_path):
    """Render spectrogram with colored component overlay."""
    disp_mask = freqs <= max_freq
    disp_db = mag_db[disp_mask, :]
    duration = float(times[-1])

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.imshow(
        disp_db, aspect="auto", origin="lower", cmap="magma",
        extent=[0, duration, 0, max_freq], interpolation="bilinear",
    )

    # Overlay components
    h = np.sum(disp_mask)
    w = len(times)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for comp in components:
        from matplotlib.colors import to_rgba
        r, g, b, _ = to_rgba(comp["color"])
        mask_disp = comp["mask"][:h, :].astype(np.float32)
        m_max = mask_disp.max()
        if m_max > 0:
            mask_norm = mask_disp / m_max
        else:
            mask_norm = mask_disp
        alpha = mask_norm * 0.35
        rgba[:, :, 0] += r * alpha
        rgba[:, :, 1] += g * alpha
        rgba[:, :, 2] += b * alpha
        rgba[:, :, 3] = np.clip(rgba[:, :, 3] + alpha, 0, 0.7)

    rgba = np.clip(rgba, 0, 1)
    ax.imshow(rgba, aspect="auto", origin="lower",
              extent=[0, duration, 0, max_freq], interpolation="nearest")

    ax.set_xlabel("Time (s)", color="#aaa")
    ax.set_ylabel("Frequency (Hz)", color="#aaa")
    ax.set_title(f"{len(components)} components", color="#ccc")
    ax.tick_params(colors="#aaa")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


async def run_decomposition(audio_path: str, method: str, params: dict) -> tuple[str, dict]:
    """Run decomposition and return (job_id, results_dict)."""
    job_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(JOBS_DIR, "isolator", job_id)
    os.makedirs(output_dir, exist_ok=True)

    sig, fs = load_audio(audio_path)
    n_fft = params["n_fft"]
    overlap_pct = params["overlap"]
    window = params["window"]
    max_freq = min(params["max_freq"], fs // 2)

    hop = max(1, int(n_fft * (1 - overlap_pct / 100)))
    freqs, times, Zxx = stft(
        sig, fs=fs, window=window,
        nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
    )
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-10)
    phase = np.angle(Zxx)

    # Run decomposition
    if method == "nmf":
        components = _decompose_nmf(mag, freqs, times, params)
    elif method == "watershed":
        components = _decompose_watershed(mag, mag_db, freqs, times, params)
    else:
        components = _decompose_cca(mag, mag_db, freqs, times, params)

    # Render spectrogram
    spec_path = os.path.join(output_dir, "spectrogram.png")
    _render_spectrogram(mag_db, freqs, times, max_freq, components, spec_path)

    # Reconstruct and export each component's audio
    comp_results = []
    for i, comp in enumerate(components):
        mask = comp["mask"]
        soft = gaussian_filter(mask, sigma=1.0)
        masked_stft = soft * mag * np.exp(1j * phase)
        _, audio = istft(
            masked_stft, fs=fs, window=window,
            nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
        )
        audio = audio[:len(sig)]

        # Save WAV
        peak = np.max(np.abs(audio))
        if peak > 1e-10:
            clip_int = np.int16(audio / peak * 32767 * 0.9)
            wav_name = f"component_{i}.wav"
            wavfile.write(os.path.join(output_dir, wav_name), fs, clip_int)
        else:
            wav_name = None

        # Save component spectrogram
        comp_spec_name = f"component_{i}.png"
        comp_mag_db = 20 * np.log10(soft * mag + 1e-10)
        _render_spectrogram(comp_mag_db, freqs, times, max_freq, [comp],
                            os.path.join(output_dir, comp_spec_name))

        comp_results.append({
            "id": i,
            "freq_peak": comp["freq_peak"],
            "energy": comp["energy"],
            "time_range": [comp["time_min"], comp["time_max"]],
            "color": comp["color"],
            "audio_url": f"/api/isolator/results/{job_id}/{wav_name}" if wav_name else None,
            "spectrogram_url": f"/api/isolator/results/{job_id}/{comp_spec_name}",
        })

    return job_id, {
        "spectrogram_url": f"/api/isolator/results/{job_id}/spectrogram.png",
        "num_components": len(components),
        "components": comp_results,
    }
