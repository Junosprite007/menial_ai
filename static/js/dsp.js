// ═══════════════════════════════════════════════════════════════
// DSP
// ═══════════════════════════════════════════════════════════════
function gW(t, sz) {
    const w = new Float64Array(sz);
    for (let n = 0; n < sz; n++) {
        switch (t) {
            case 'rectangular':
                w[n] = 1;
                break;
            case 'hann':
                w[n] =
                    0.5 *
                    (1 - Math.cos((2 * Math.PI * n) / (sz - 1)));
                break;
            case 'hamming':
                w[n] =
                    0.54 -
                    0.46 * Math.cos((2 * Math.PI * n) / (sz - 1));
                break;
            case 'blackman':
                w[n] =
                    0.42 -
                    0.5 * Math.cos((2 * Math.PI * n) / (sz - 1)) +
                    0.08 * Math.cos((4 * Math.PI * n) / (sz - 1));
                break;
        }
    }
    return w;
}
function fI(re, im) {
    const n = re.length;
    if (n <= 1) return;
    for (let i = 1, j = 0; i < n; i++) {
        let b = n >> 1;
        for (; j & b; b >>= 1) j ^= b;
        j ^= b;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    for (let l = 2; l <= n; l <<= 1) {
        const a = (-2 * Math.PI) / l,
            wR = Math.cos(a),
            wI = Math.sin(a);
        for (let i = 0; i < n; i += l) {
            let cR = 1,
                cI = 0;
            for (let j = 0; j < l / 2; j++) {
                const uR = re[i + j],
                    uI = im[i + j],
                    vR =
                        re[i + j + l / 2] * cR -
                        im[i + j + l / 2] * cI,
                    vI =
                        re[i + j + l / 2] * cI +
                        im[i + j + l / 2] * cR;
                re[i + j] = uR + vR;
                im[i + j] = uI + vI;
                re[i + j + l / 2] = uR - vR;
                im[i + j + l / 2] = uI - vI;
                const nR = cR * wR - cI * wI;
                cI = cR * wI + cI * wR;
                cR = nR;
            }
        }
    }
}
function ifI(re, im) {
    const n = re.length;
    for (let i = 0; i < n; i++) im[i] = -im[i];
    fI(re, im);
    for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] = -im[i] / n;
    }
}

function fftSlice(sig, st, sz, wt, zp) {
    let n = 1;
    while (n < sz * zp) n <<= 1;
    const w = gW(wt, sz),
        re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < sz; i++) {
        const idx = st + i;
        if (idx >= 0 && idx < sig.length) re[i] = sig[idx] * w[i];
    }
    fI(re, im);
    const h = n / 2 + 1,
        m = new Float64Array(h);
    for (let i = 0; i < h; i++)
        m[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    return { mag: m, n };
}
function fullFFT(sig, wt, sr, zp) {
    let n = 1;
    while (n < sig.length * zp) n <<= 1;
    const w = gW(wt, sig.length),
        re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < sig.length; i++) re[i] = sig[i] * w[i];
    fI(re, im);
    const h = n / 2 + 1,
        m = new Float64Array(h),
        f = new Float64Array(h);
    for (let i = 0; i < h; i++) {
        m[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
        f[i] = (i * sr) / n;
    }
    return { mag: m, freqs: f, n };
}
function compSTFT(sig, nf, hp, wt, zp) {
    const w = gW(wt, nf);
    let fs = 1;
    while (fs < nf * zp) fs <<= 1;
    const nfr = Math.max(1, Math.floor((sig.length - nf) / hp) + 1),
        nb = fs / 2 + 1,
        sp = [];
    for (let f = 0; f < nfr; f++) {
        const s = f * hp,
            re = new Float64Array(fs),
            im = new Float64Array(fs);
        for (let i = 0; i < nf && s + i < sig.length; i++)
            re[i] = sig[s + i] * w[i];
        fI(re, im);
        const m = new Float64Array(nb);
        for (let i = 0; i < nb; i++)
            m[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
        sp.push(m);
    }
    return { spec: sp, numBins: nb, numFrames: nfr, fftSize: fs };
}

// ── BANDPASS FILTER via FFT (legacy brick-wall) ──
function bandpassFilter(sig, sr, loHz, hiHz) {
    let n = 1;
    while (n < sig.length) n <<= 1;
    const re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < sig.length; i++) re[i] = sig[i];
    fI(re, im);
    const binLo = Math.floor((loHz * n) / sr),
        binHi = Math.ceil((hiHz * n) / sr);
    for (let i = 0; i < n; i++) {
        if (i <= n / 2) {
            if (i < binLo || i > binHi) {
                re[i] = 0;
                im[i] = 0;
            }
        } else {
            const mirror = n - i;
            if (mirror < binLo || mirror > binHi) {
                re[i] = 0;
                im[i] = 0;
            }
        }
    }
    ifI(re, im);
    const out = new Float64Array(sig.length);
    for (let i = 0; i < sig.length; i++) out[i] = re[i];
    return out;
}

// ── BUTTERWORTH GAIN ──
function butterworthGain(f, fc, order, isHighpass) {
    if (isHighpass) {
        if (f <= 0) return 0;
        return 1 / Math.sqrt(1 + Math.pow(fc / f, 2 * order));
    } else {
        if (fc <= 0) return 0;
        return 1 / Math.sqrt(1 + Math.pow(f / fc, 2 * order));
    }
}

// ── SMOOTH BANDPASS (Butterworth) ──
function smoothBandpass(sig, sr, loHz, hiHz, order) {
    let n = 1;
    while (n < sig.length) n <<= 1;
    const re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < sig.length; i++) re[i] = sig[i];
    fI(re, im);
    for (let i = 0; i <= n / 2; i++) {
        const freq = (i * sr) / n;
        let gain = 1;
        if (loHz > 0)
            gain *= butterworthGain(freq, loHz, order, true);
        if (hiHz < sr / 2)
            gain *= butterworthGain(freq, hiHz, order, false);
        re[i] *= gain;
        im[i] *= gain;
        if (i > 0 && i < n / 2) {
            re[n - i] *= gain;
            im[n - i] *= gain;
        }
    }
    ifI(re, im);
    const out = new Float64Array(sig.length);
    for (let i = 0; i < sig.length; i++) out[i] = re[i];
    return out;
}

// ── MULTI-BAND FILTER ──
function multiBandFilter(sig, sr, bands) {
    if (!bands.length) return sig.slice();
    let n = 1;
    while (n < sig.length) n <<= 1;
    const re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < sig.length; i++) re[i] = sig[i];
    fI(re, im);
    const outRe = new Float64Array(n),
        outIm = new Float64Array(n);
    for (let i = 0; i <= n / 2; i++) {
        const freq = (i * sr) / n;
        let maxGain = 0;
        for (const b of bands) {
            let g = 1;
            if (b.lo > 0)
                g *= butterworthGain(
                    freq,
                    b.lo,
                    b.rolloff || 4,
                    true,
                );
            if (b.hi < sr / 2)
                g *= butterworthGain(
                    freq,
                    b.hi,
                    b.rolloff || 4,
                    false,
                );
            if (g > maxGain) maxGain = g;
        }
        outRe[i] = re[i] * maxGain;
        outIm[i] = im[i] * maxGain;
        if (i > 0 && i < n / 2) {
            outRe[n - i] = re[n - i] * maxGain;
            outIm[n - i] = im[n - i] * maxGain;
        }
    }
    ifI(outRe, outIm);
    const out = new Float64Array(sig.length);
    for (let i = 0; i < sig.length; i++) out[i] = outRe[i];
    return out;
}

// ── SPECTRAL SUBTRACTION ──
function captureNoise() {
    if (!S.sig) return;
    const s = Math.floor(S.noiseStart * S.sig.length);
    const e = Math.floor(S.noiseEnd * S.sig.length);
    const len = e - s;
    if (len < 256) {
        alert(
            'Select a longer noise region (at least 256 samples)',
        );
        return;
    }
    let n = 1;
    while (n < len) n <<= 1;
    const re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < len; i++) re[i] = S.sig[s + i];
    fI(re, im);
    const profile = new Float64Array(n / 2 + 1);
    for (let i = 0; i <= n / 2; i++)
        profile[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    S.noiseProfile = profile;
    S.noiseFFTSize = n;
    const st = $('noiseStatus');
    if (st) st.textContent = 'Captured (' + len + ' samples)';
    const cs = $('cNoiseStatus');
    if (cs) cs.textContent = 'Captured (' + len + ' samples)';
    recomputeExtraction();
    if (!S.playing) rAll();
}

function spectralSubtract(
    sig,
    sr,
    noiseProfile,
    noiseFftSize,
    alpha,
    beta,
) {
    let n = 1;
    while (n < sig.length) n <<= 1;
    const re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < sig.length; i++) re[i] = sig[i];
    fI(re, im);
    const halfN = n / 2;
    const halfNoise = noiseFftSize / 2;
    for (let i = 0; i <= halfN; i++) {
        const mag = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
        const phase = Math.atan2(im[i], re[i]);
        const nIdx = Math.min(
            Math.round((i / halfN) * halfNoise),
            noiseProfile.length - 1,
        );
        const noiseMag = noiseProfile[nIdx] || 0;
        let cleanMag = mag - alpha * noiseMag;
        cleanMag = Math.max(cleanMag, beta * mag);
        re[i] = cleanMag * Math.cos(phase);
        im[i] = cleanMag * Math.sin(phase);
        if (i > 0 && i < halfN) {
            re[n - i] = cleanMag * Math.cos(-phase);
            im[n - i] = cleanMag * Math.sin(-phase);
        }
    }
    ifI(re, im);
    const out = new Float64Array(sig.length);
    for (let i = 0; i < sig.length; i++) out[i] = re[i];
    return out;
}

// ── Fourier Series ──
function compFS(sig, st, ws) {
    const chunk = new Float64Array(ws);
    for (let i = 0; i < ws; i++) {
        const idx = st + i;
        chunk[i] = idx >= 0 && idx < sig.length ? sig[idx] : 0;
    }
    let n = 1;
    while (n < ws) n <<= 1;
    const re = new Float64Array(n),
        im = new Float64Array(n);
    for (let i = 0; i < ws; i++) re[i] = chunk[i];
    fI(re, im);
    const mH = Math.min(Math.floor(ws / 2), 256),
        h = [];
    for (let k = 0; k <= mH; k++) {
        const amp =
            k === 0
                ? Math.sqrt(re[0] * re[0] + im[0] * im[0]) / ws
                : (2 * Math.sqrt(re[k] * re[k] + im[k] * im[k])) /
                  ws;
        h.push({ k, amp, phase: Math.atan2(im[k], re[k]) });
    }
    // Build reconstruction audio for playback
    return { harmonics: h, chunk, re, im, n: n };
}

function reconstructFS(harmonics, ws, numH) {
    const out = new Float64Array(ws);
    for (let k = 0; k <= numH && k < harmonics.length; k++) {
        const h = harmonics[k];
        for (let i = 0; i < ws; i++) {
            const t = i / ws;
            out[i] +=
                k === 0
                    ? h.amp * Math.cos(h.phase)
                    : h.amp *
                      Math.cos(2 * Math.PI * k * t + h.phase);
        }
    }
    return out;
}
