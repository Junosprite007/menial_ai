// ═══════════════════════════════════════════════════════════════
// ANIMATION
// ═══════════════════════════════════════════════════════════════
function startAL() {
    function lp() {
        if (!S.playing) return;
        rFr();
        S.aF = requestAnimationFrame(lp);
    }
    S.aF = requestAnimationFrame(lp);
}
function stopAL() {
    if (S.aF) {
        cancelAnimationFrame(S.aF);
        S.aF = null;
    }
}
function rFr() {
    const t = curT(),
        f = S.dur > 0 ? t / S.dur : 0;
    $('seek').value = Math.round(f * 1000);
    $('tDisp').textContent = fmtT(t) + ' / ' + fmtT(S.dur);
    dWave(f);
    if (S.tab === 'fs') dFS();
    else if (S.anOn) {
        S.tab === 'fft' ? dFFT(f) : dSTFT(f);
        if (S.playing) dLive(t);
    } else dOff();
}
function rAll() {
    const f = S.dur > 0 ? S.pOff / S.dur : 0;
    $('seek').value = Math.round(f * 1000);
    $('tDisp').textContent = fmtT(S.pOff) + ' / ' + fmtT(S.dur);
    dWave(f);
    if (S.tab === 'fs') dFS();
    else if (S.anOn) {
        S.tab === 'fft' ? dFFT(f) : dSTFT(f);
    } else dOff();
    $('livS').style.display =
        S.playing && S.anOn && S.tab !== 'fs' ? '' : 'none';
    updP();
}

// ═══════════════════════════════════════════════════════════════
// DRAWING
// ═══════════════════════════════════════════════════════════════
function gC(id, h) {
    const c = $(id),
        dp = devicePixelRatio || 1,
        r = c.parentElement.getBoundingClientRect(),
        W = r.width;
    c.width = W * dp;
    c.height = h * dp;
    c.style.height = h + 'px';
    const cx = c.getContext('2d');
    cx.scale(dp, dp);
    return { cx, W, H: h };
}

function dWave(pf) {
    const { cx, W, H } = gC('wC', 110);
    cx.fillStyle = '#0a0a0f';
    cx.fillRect(0, 0, W, H);
    if (!S.sig) return;
    const sig = S.sig;
    // FS window highlight
    if (S.tab === 'fs') {
        const off = Math.floor(
            (S.fsOff / 1000) * Math.max(0, sig.length - S.fsW),
        );
        const x0 = (off / sig.length) * W,
            x1 = ((off + S.fsW) / sig.length) * W;
        cx.fillStyle = 'rgba(251,191,36,.08)';
        cx.fillRect(x0, 0, x1 - x0, H);
        cx.strokeStyle = 'rgba(251,191,36,.3)';
        cx.lineWidth = 1;
        cx.strokeRect(x0, 0, x1 - x0, H);
    }
    // Noise region highlight
    if (
        (S.isoMode === 'spectral' || S.isoMode === 'combined') &&
        S.tab !== 'fs'
    ) {
        const nx0 = S.noiseStart * W,
            nx1 = S.noiseEnd * W;
        cx.fillStyle = 'rgba(251,191,36,.08)';
        cx.fillRect(nx0, 0, nx1 - nx0, H);
        cx.strokeStyle = 'rgba(251,191,36,.25)';
        cx.lineWidth = 1;
        cx.setLineDash([3, 3]);
        cx.strokeRect(nx0, 0, nx1 - nx0, H);
        cx.setLineDash([]);
        cx.font = '8px "DM Mono",monospace';
        cx.fillStyle = 'rgba(251,191,36,.5)';
        cx.textAlign = 'center';
        cx.fillText('noise', (nx0 + nx1) / 2, 10);
    }
    // Original waveform
    cx.beginPath();
    cx.strokeStyle = '#3b82f6';
    cx.lineWidth = 0.7;
    const step = Math.max(1, Math.floor(sig.length / W));
    for (let px = 0; px < W; px++) {
        const idx = Math.floor((px / W) * sig.length);
        let mn = 1e9,
            mx = -1e9;
        for (let j = 0; j < step && idx + j < sig.length; j++) {
            mn = Math.min(mn, sig[idx + j]);
            mx = Math.max(mx, sig[idx + j]);
        }
        const y1 = H / 2 - mx * H * 0.44,
            y2 = H / 2 - mn * H * 0.44;
        px === 0 ? cx.moveTo(px, y1) : 0;
        cx.lineTo(px, y1);
        cx.lineTo(px, y2);
    }
    cx.stroke();
    // Extraction overlay
    const ext = S.tab === 'fs' ? S.fsSig : S.extSig;
    if (S.showOv && ext && ext.length === sig.length) {
        cx.beginPath();
        cx.strokeStyle = 'rgba(34,211,238,.65)';
        cx.lineWidth = 0.9;
        for (let px = 0; px < W; px++) {
            const idx = Math.floor((px / W) * ext.length);
            let mn = 1e9,
                mx = -1e9;
            for (let j = 0; j < step && idx + j < ext.length; j++) {
                mn = Math.min(mn, ext[idx + j]);
                mx = Math.max(mx, ext[idx + j]);
            }
            const y1 = H / 2 - mx * H * 0.44,
                y2 = H / 2 - mn * H * 0.44;
            px === 0 ? cx.moveTo(px, y1) : 0;
            cx.lineTo(px, y1);
            cx.lineTo(px, y2);
        }
        cx.stroke();
    }
    // Center line
    cx.strokeStyle = 'rgba(255,255,255,.05)';
    cx.lineWidth = 0.5;
    cx.beginPath();
    cx.moveTo(0, H / 2);
    cx.lineTo(W, H / 2);
    cx.stroke();
    // Selection region
    if (S.hasSelection) {
        const sx0 = S.selStart * W,
            sx1 = S.selEnd * W;
        // Dim areas outside selection
        cx.fillStyle = 'rgba(0,0,0,.45)';
        cx.fillRect(0, 0, sx0, H);
        cx.fillRect(sx1, 0, W - sx1, H);
        // Selection edges
        cx.strokeStyle = 'rgba(52,211,153,.7)';
        cx.lineWidth = 1.5;
        cx.beginPath();
        cx.moveTo(sx0, 0);
        cx.lineTo(sx0, H);
        cx.stroke();
        cx.beginPath();
        cx.moveTo(sx1, 0);
        cx.lineTo(sx1, H);
        cx.stroke();
        // Selection label
        cx.font = '8px "DM Mono",monospace';
        cx.fillStyle = 'rgba(52,211,153,.6)';
        cx.textAlign = 'center';
        const selDur =
            (S.selEnd - S.selStart) * (sig.length / S.sr);
        cx.fillText(
            selDur.toFixed(2) + 's selected',
            (sx0 + sx1) / 2,
            10,
        );
    }
    // Playhead
    if (pf >= 0 && S.tab !== 'fs') {
        const x = pf * W;
        cx.strokeStyle = S.playing
            ? S.playExt
                ? '#22d3ee'
                : '#34d399'
            : 'rgba(167,139,250,.6)';
        cx.lineWidth = S.playing ? 2 : 1.5;
        cx.beginPath();
        cx.moveTo(x, 0);
        cx.lineTo(x, H);
        cx.stroke();
    }
    // Legend
    cx.font = '9px "DM Mono",monospace';
    cx.textAlign = 'left';
    cx.fillStyle = '#3b82f6';
    cx.fillText('■ original', 6, H - 4);
    if (S.showOv && ext) {
        cx.fillStyle = '#22d3ee';
        cx.fillText('■ extracted', 70, H - 4);
    }
    if (S.hasSelection) {
        cx.fillStyle = '#34d399';
        cx.fillText(
            '■ selected',
            S.showOv && ext ? 150 : 70,
            H - 4,
        );
    }
    $('wCap').textContent =
        `${S.lbl} · ${sig.length.toLocaleString()} · ${S.sr} Hz · ${(sig.length / S.sr).toFixed(2)}s`;
    // Selection detail caption
    const selCapEl = $('selCap');
    if (selCapEl) {
        if (S.hasSelection) {
            const sT = ((S.selStart * sig.length) / S.sr).toFixed(
                3,
            );
            const eT = ((S.selEnd * sig.length) / S.sr).toFixed(3);
            const dur = (
                ((S.selEnd - S.selStart) * sig.length) /
                S.sr
            ).toFixed(3);
            selCapEl.textContent = `Selection: ${sT}s — ${eT}s (${dur}s) · Drag waveform to select · Double-click to clear`;
        } else {
            selCapEl.textContent =
                'Drag on waveform to select a region for playback';
        }
    }
}

function dFFT(pf) {
    $('anOff').style.display = 'none';
    const { cx, W, H } = gC('mC', 280);
    cx.fillStyle = '#0a0a0f';
    cx.fillRect(0, 0, W, H);
    if (!S.sig) return;
    let mag, freqs, n;
    if (pf > 0 && pf < 1) {
        const c = Math.floor(pf * S.sig.length),
            st = Math.max(0, c - Math.floor(S.nfft / 2));
        const r = fftSlice(S.sig, st, S.nfft, S.wt, S.zpR);
        mag = r.mag;
        n = r.n;
        freqs = new Float64Array(mag.length);
        for (let i = 0; i < mag.length; i++)
            freqs[i] = (i * S.sr) / n;
    } else {
        const r = fullFFT(S.sig, S.wt, S.sr, S.zpR);
        mag = r.mag;
        freqs = r.freqs;
        n = r.n;
    }
    const dm = Math.min(S.mf, S.sr / 2);
    let mxD = -1e9;
    const mD = [];
    for (let i = 0; i < mag.length; i++) {
        const d = 20 * Math.log10(mag[i] + 1e-10);
        if (d > mxD) mxD = d;
        mD.push(d);
    }
    const mnD = mxD - S.dynR;
    const p = { l: 48, r: 12, t: 12, b: 30 },
        pW = W - p.l - p.r,
        pH = H - p.t - p.b;
    // Band highlight(s)
    if (S.isoMode === 'multi' || S.isoMode === 'combined') {
        S.bands.forEach((b, idx) => {
            const col = HC[idx % HC.length];
            const bx0 = p.l + (b.lo / dm) * pW,
                bx1 = p.l + (Math.min(b.hi, dm) / dm) * pW;
            cx.fillStyle = col + '12';
            cx.fillRect(
                Math.max(bx0, p.l),
                p.t,
                Math.min(bx1, p.l + pW) - Math.max(bx0, p.l),
                pH,
            );
            cx.strokeStyle = col + '66';
            cx.lineWidth = 1;
            cx.setLineDash([4, 4]);
            if (b.lo > 0 && b.lo <= dm) {
                cx.beginPath();
                cx.moveTo(bx0, p.t);
                cx.lineTo(bx0, p.t + pH);
                cx.stroke();
            }
            if (b.hi < dm) {
                cx.beginPath();
                cx.moveTo(bx1, p.t);
                cx.lineTo(bx1, p.t + pH);
                cx.stroke();
            }
            cx.setLineDash([]);
        });
    } else if (S.isoMode === 'single') {
        const bx0 = p.l + (S.bLo / dm) * pW,
            bx1 = p.l + (Math.min(S.bHi, dm) / dm) * pW;
        cx.fillStyle = 'rgba(34,211,238,.06)';
        cx.fillRect(
            Math.max(bx0, p.l),
            p.t,
            Math.min(bx1, p.l + pW) - Math.max(bx0, p.l),
            pH,
        );
        cx.strokeStyle = 'rgba(34,211,238,.3)';
        cx.lineWidth = 1;
        cx.setLineDash([4, 4]);
        if (S.bLo > 0 && S.bLo <= dm) {
            cx.beginPath();
            cx.moveTo(bx0, p.t);
            cx.lineTo(bx0, p.t + pH);
            cx.stroke();
        }
        if (S.bHi < dm) {
            cx.beginPath();
            cx.moveTo(bx1, p.t);
            cx.lineTo(bx1, p.t + pH);
            cx.stroke();
        }
        cx.setLineDash([]);
        // Butterworth gain curve overlay
        cx.beginPath();
        cx.strokeStyle = 'rgba(251,191,36,.4)';
        cx.lineWidth = 1.5;
        for (let px = 0; px < pW; px++) {
            const freq = (px / pW) * dm;
            let gain = 1;
            if (S.bLo > 0)
                gain *= butterworthGain(
                    freq,
                    S.bLo,
                    S.rolloff,
                    true,
                );
            if (S.bHi < S.sr / 2)
                gain *= butterworthGain(
                    freq,
                    S.bHi,
                    S.rolloff,
                    false,
                );
            const y = p.t + pH - gain * pH;
            px === 0
                ? cx.moveTo(p.l + px, y)
                : cx.lineTo(p.l + px, y);
        }
        cx.stroke();
    }
    // Grid
    cx.strokeStyle = 'rgba(255,255,255,.04)';
    cx.lineWidth = 0.5;
    for (let f = 0; f <= dm; f += dm > 4e3 ? 1e3 : 500) {
        const x = p.l + (f / dm) * pW;
        cx.beginPath();
        cx.moveTo(x, p.t);
        cx.lineTo(x, p.t + pH);
        cx.stroke();
    }
    // FFT
    cx.beginPath();
    cx.strokeStyle = '#a78bfa';
    cx.lineWidth = 1;
    for (let i = 0; i < mag.length; i++) {
        if (freqs[i] > dm) break;
        const x = p.l + (freqs[i] / dm) * pW,
            nm = Math.max(0, (mD[i] - mnD) / (mxD - mnD)),
            y = p.t + pH - nm * pH;
        i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
    }
    cx.stroke();
    // Axis
    cx.fillStyle = 'rgba(255,255,255,.3)';
    cx.font = '9px "DM Mono",monospace';
    cx.textAlign = 'center';
    for (let f = 0; f <= dm; f += dm > 4e3 ? 2e3 : 1e3)
        cx.fillText(f + '', p.l + (f / dm) * pW, H - 6);
    cx.textAlign = 'right';
    cx.fillText(mxD.toFixed(0) + 'dB', p.l - 3, p.t + 10);
    // Band labels
    cx.font = '9px "DM Mono",monospace';
    cx.textAlign = 'center';
    if (S.isoMode === 'single') {
        const bx0 = p.l + (S.bLo / dm) * pW,
            bx1 = p.l + (Math.min(S.bHi, dm) / dm) * pW;
        const bcx =
            (Math.max(bx0, p.l) + Math.min(bx1, p.l + pW)) / 2;
        cx.fillStyle = 'rgba(34,211,238,.5)';
        cx.fillText(
            `extracted: ${S.bLo}-${Math.min(S.bHi, Math.floor(dm))} Hz (order ${S.rolloff})`,
            bcx,
            p.t + 12,
        );
    } else if (S.isoMode === 'multi' || S.isoMode === 'combined') {
        cx.fillStyle = 'rgba(167,139,250,.5)';
        cx.fillText(
            `${S.bands.length} band${S.bands.length !== 1 ? 's' : ''} · ${S.isoMode}`,
            p.l + pW / 2,
            p.t + 12,
        );
    } else if (S.isoMode === 'spectral') {
        cx.fillStyle = 'rgba(251,191,36,.5)';
        cx.fillText(
            S.noiseProfile
                ? `spectral subtraction (α=${S.overSub.toFixed(1)})`
                : 'spectral sub — capture noise first',
            p.l + pW / 2,
            p.t + 12,
        );
    }
}

let sIC = null,
    sCK = '';
function dSTFT(pf) {
    $('anOff').style.display = 'none';
    const { cx, W, H } = gC('mC', 300);
    cx.fillStyle = '#0a0a0f';
    cx.fillRect(0, 0, W, H);
    if (!S.sig) return;
    const hp = Math.max(1, Math.floor(S.nfft * (1 - S.olP / 100))),
        ck = `${S.sig.length}_${S.nfft}_${hp}_${S.wt}_${S.mf}_${S.dynR}_${S.zpR}_${W}_${H}`;
    const p = { l: 42, r: 8, t: 8, b: 26 },
        pW = W - p.l - p.r,
        pH = H - p.t - p.b,
        dp = devicePixelRatio || 1;
    if (sCK !== ck) {
        const { spec, numBins, numFrames, fftSize } = compSTFT(
            S.sig,
            S.nfft,
            hp,
            S.wt,
            S.zpR,
        );
        S.stD = { spec, numBins, numFrames, fftSize, hop: hp };
        const ny = S.sr / 2,
            dm = Math.min(S.mf, ny),
            mbi = Math.floor((dm / ny) * (numBins - 1));
        let gM = -1e9;
        const db = spec.map(f =>
            f.map(v => {
                const d = 20 * Math.log10(v + 1e-10);
                if (d > gM) gM = d;
                return d;
            }),
        );
        const gm = gM - S.dynR;
        const iW = Math.ceil(pW * dp),
            iH = Math.ceil(pH * dp),
            id = cx.createImageData(iW, iH);
        for (let px = 0; px < iW; px++) {
            const fi = Math.min(
                    Math.floor((px / iW) * numFrames),
                    numFrames - 1,
                ),
                fr = db[fi];
            for (let py = 0; py < iH; py++) {
                const bi = Math.min(
                        Math.floor(((iH - py) / iH) * mbi),
                        numBins - 1,
                    ),
                    v = (fr[bi] - gm) / S.dynR;
                const [r, g, b] = mc(v);
                const i = (py * iW + px) * 4;
                id.data[i] = r;
                id.data[i + 1] = g;
                id.data[i + 2] = b;
                id.data[i + 3] = 255;
            }
        }
        sIC = id;
        sCK = ck;
    }
    if (sIC) cx.putImageData(sIC, p.l * dp, p.t * dp);
    // Band overlay
    const ny = S.sr / 2,
        dm = Math.min(S.mf, ny);
    function drawSTFTBand(lo, hi, color) {
        const by0 = p.t + pH - (Math.min(hi, dm) / dm) * pH,
            by1 = p.t + pH - (lo / dm) * pH;
        cx.fillStyle = color + '1a';
        cx.fillRect(
            p.l,
            Math.max(by0, p.t),
            pW,
            Math.min(by1, p.t + pH) - Math.max(by0, p.t),
        );
        cx.strokeStyle = color + '66';
        cx.lineWidth = 1;
        cx.setLineDash([4, 3]);
        if (lo > 0) {
            cx.beginPath();
            cx.moveTo(p.l, by1);
            cx.lineTo(p.l + pW, by1);
            cx.stroke();
        }
        if (hi < dm) {
            cx.beginPath();
            cx.moveTo(p.l, by0);
            cx.lineTo(p.l + pW, by0);
            cx.stroke();
        }
        cx.setLineDash([]);
    }
    if (S.isoMode === 'multi' || S.isoMode === 'combined') {
        S.bands.forEach((b, idx) =>
            drawSTFTBand(b.lo, b.hi, HC[idx % HC.length]),
        );
    } else if (S.isoMode === 'single') {
        drawSTFTBand(S.bLo, S.bHi, '#22d3ee');
    }
    // Playhead
    if (pf >= 0) {
        const x = p.l + pf * pW;
        cx.strokeStyle = S.playing
            ? S.playExt
                ? '#22d3ee'
                : '#34d399'
            : 'rgba(167,139,250,.6)';
        cx.lineWidth = S.playing ? 2 : 1.5;
        cx.beginPath();
        cx.moveTo(x, p.t);
        cx.lineTo(x, p.t + pH);
        cx.stroke();
    }
    // Axis
    cx.fillStyle = 'rgba(255,255,255,.3)';
    cx.font = '9px "DM Mono",monospace';
    cx.textAlign = 'right';
    for (let f = 0; f <= dm; f += dm > 4e3 ? 2e3 : 1e3) {
        const y = p.t + pH - (f / dm) * pH;
        cx.fillText(f + '', p.l - 3, y + 3);
    }
    cx.textAlign = 'center';
    const dur = S.sig.length / S.sr;
    for (
        let s = 0;
        s <= dur;
        s += dur > 3 ? 1 : dur > 1 ? 0.5 : 0.2
    ) {
        const x = p.l + (s / dur) * pW;
        cx.fillText(s.toFixed(1) + 's', x, H - 4);
    }
}

function dLive(ct) {
    if (!S.sig || !S.anOn) return;
    const { cx, W, H } = gC('livC', 150);
    cx.fillStyle = '#0a0a0f';
    cx.fillRect(0, 0, W, H);
    const c = Math.floor((ct / S.dur) * S.sig.length),
        st = Math.max(0, c - Math.floor(S.nfft / 2));
    const { mag, n } = fftSlice(S.sig, st, S.nfft, S.wt, S.zpR);
    const dm = Math.min(S.mf, S.sr / 2);
    let mxD = -1e9;
    const mD = [];
    for (let i = 0; i < mag.length; i++) {
        const d = 20 * Math.log10(mag[i] + 1e-10);
        if (d > mxD) mxD = d;
        mD.push(d);
    }
    const mnD = mxD - S.dynR;
    const p = { l: 42, r: 8, t: 8, b: 22 },
        pW = W - p.l - p.r,
        pH = H - p.t - p.b;
    const nb = Math.floor((dm / (S.sr / 2)) * mag.length),
        bW = Math.max(1, pW / nb);
    for (let i = 0; i < nb && i < mag.length; i++) {
        const nm = Math.max(0, (mD[i] - mnD) / (mxD - mnD)),
            x = p.l + (i / nb) * pW,
            bH = nm * pH;
        const freq = (i * S.sr) / n;
        let inBand = false;
        if (S.isoMode === 'single') {
            inBand = freq >= S.bLo && freq <= S.bHi;
        } else if (
            S.isoMode === 'multi' ||
            S.isoMode === 'combined'
        ) {
            inBand = S.bands.some(
                b => freq >= b.lo && freq <= b.hi,
            );
        }
        if (inBand) {
            cx.fillStyle = `rgba(34,211,238,${0.4 + nm * 0.6})`;
        } else {
            const h = 260 - nm * 80;
            cx.fillStyle = `hsla(${h},50%,${40 + nm * 15}%,${0.3 + nm * 0.3})`;
        }
        cx.fillRect(x, p.t + pH - bH, Math.max(bW - 0.5, 0.5), bH);
    }
}

function dOff() {
    const { cx, W, H } = gC('mC', S.tab === 'stft' ? 300 : 280);
    cx.fillStyle = '#0a0a0f';
    cx.fillRect(0, 0, W, H);
    $('anOff').style.display = '';
}

// ═══════════════════════════════════════════════════════════════
// FOURIER SERIES
// ═══════════════════════════════════════════════════════════════
function dFS() {
    $('anOff').style.display = 'none';
    if (!S.sig) return;
    const off = Math.floor(
        (S.fsOff / 1000) * Math.max(0, S.sig.length - S.fsW),
    );
    const { harmonics, chunk } = compFS(S.sig, off, S.fsW);
    const numH = Math.min(S.fsH, harmonics.length - 1);
    // Main: tiled window
    const { cx, W, H } = gC('mC', 200);
    cx.fillStyle = '#0a0a0f';
    cx.fillRect(0, 0, W, H);
    const p = { l: 6, r: 6, t: 12, b: 18 },
        pW = W - p.l - p.r,
        pH = H - p.t - p.b;
    let mn = 1e9,
        mx = -1e9;
    for (let i = 0; i < chunk.length; i++) {
        if (chunk[i] < mn) mn = chunk[i];
        if (chunk[i] > mx) mx = chunk[i];
    }
    const amp = Math.max(Math.abs(mn), Math.abs(mx), 0.001);
    for (let cp = -1; cp <= 1; cp++) {
        const xO = p.l + ((cp + 1) / 3) * pW,
            sW = pW / 3;
        cx.strokeStyle =
            cp === 0
                ? 'rgba(255,255,255,.6)'
                : 'rgba(255,255,255,.12)';
        cx.lineWidth = cp === 0 ? 1.2 : 0.5;
        if (cp === 0) {
            cx.fillStyle = 'rgba(255,255,255,.02)';
            cx.fillRect(xO, p.t, sW, pH);
        }
        cx.beginPath();
        for (let i = 0; i < chunk.length; i++) {
            const x = xO + (i / chunk.length) * sW,
                y = p.t + pH / 2 - (chunk[i] / amp) * pH * 0.42;
            i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
        }
        cx.stroke();
    }
    cx.fillStyle = 'rgba(251,191,36,.4)';
    cx.font = '10px "DM Mono",monospace';
    cx.textAlign = 'center';
    cx.fillText(
        `f₁ = ${(S.sr / S.fsW).toFixed(1)} Hz · Period = ${((S.fsW / S.sr) * 1e3).toFixed(1)} ms`,
        W / 2,
        p.t - 1,
    );

    // Recon
    const recon = reconstructFS(harmonics, S.fsW, numH);
    const { cx: rx, W: rW, H: rH } = gC('fsRC', 90);
    rx.fillStyle = '#0a0a0f';
    rx.fillRect(0, 0, rW, rH);
    const rp = { l: 6, r: 6, t: 6, b: 6 },
        rpW = rW - rp.l - rp.r,
        rpH = rH - rp.t - rp.b;
    rx.strokeStyle = 'rgba(255,255,255,.25)';
    rx.lineWidth = 1;
    rx.beginPath();
    for (let i = 0; i < chunk.length; i++) {
        const x = rp.l + (i / chunk.length) * rpW,
            y = rp.t + rpH / 2 - (chunk[i] / amp) * rpH * 0.42;
        i === 0 ? rx.moveTo(x, y) : rx.lineTo(x, y);
    }
    rx.stroke();
    rx.strokeStyle = '#34d399';
    rx.lineWidth = 1.5;
    rx.beginPath();
    for (let i = 0; i < recon.length; i++) {
        const x = rp.l + (i / recon.length) * rpW,
            y = rp.t + rpH / 2 - (recon[i] / amp) * rpH * 0.42;
        i === 0 ? rx.moveTo(x, y) : rx.lineTo(x, y);
    }
    rx.stroke();
    rx.font = '9px "DM Mono",monospace';
    rx.textAlign = 'left';
    rx.fillStyle = 'rgba(255,255,255,.3)';
    rx.fillText('white=original', 5, 12);
    rx.fillStyle = 'rgba(52,211,153,.7)';
    rx.fillText('green=reconstruction', 110, 12);
    $('fsRCap').textContent =
        `${numH} harmonics of ${Math.floor(S.fsW / 2)} possible`;

    // Harmonics
    const sc = $('fsScr');
    const need = numH + 1;
    while (sc.children.length > need) sc.removeChild(sc.lastChild);
    while (sc.children.length < need) {
        const r = document.createElement('div');
        r.className = 'fhr';
        const l = document.createElement('div');
        l.className = 'fhl';
        const c = document.createElement('canvas');
        c.className = 'fhc';
        c.height = 32;
        r.appendChild(l);
        r.appendChild(c);
        sc.appendChild(r);
    }
    for (let k = 0; k <= numH; k++) {
        const h = harmonics[k],
            row = sc.children[k],
            lbl = row.children[0],
            cvs = row.children[1];
        const freq =
            k === 0
                ? 'DC'
                : `${((k * S.sr) / S.fsW).toFixed(0)} Hz`;
        const col = HC[k % HC.length];
        lbl.innerHTML = `<strong>k=${k}</strong> ${freq}<br><span style="font-size:8px;color:${col}">amp ${h.amp.toFixed(3)}</span>`;
        const dp = devicePixelRatio || 1,
            cW = Math.max(
                100,
                cvs.parentElement.getBoundingClientRect().width -
                    110,
            );
        cvs.style.width = cW + 'px';
        cvs.width = cW * dp;
        cvs.height = 32 * dp;
        const cc = cvs.getContext('2d');
        cc.scale(dp, dp);
        cc.fillStyle = 'rgba(0,0,0,.3)';
        cc.fillRect(0, 0, cW, 32);
        cc.strokeStyle = 'rgba(255,255,255,.04)';
        cc.lineWidth = 0.5;
        cc.beginPath();
        cc.moveTo(0, 16);
        cc.lineTo(cW, 16);
        cc.stroke();
        cc.strokeStyle = col;
        cc.lineWidth = 1.2;
        cc.beginPath();
        for (let i = 0; i < S.fsW; i++) {
            const t = i / S.fsW,
                x = (i / S.fsW) * cW;
            const v =
                k === 0
                    ? h.amp * Math.cos(h.phase)
                    : h.amp *
                      Math.cos(2 * Math.PI * k * t + h.phase);
            const y = 16 - (v / amp) * 14;
            i === 0 ? cc.moveTo(x, y) : cc.lineTo(x, y);
        }
        cc.stroke();
    }
}

// ═══════════════════════════════════════════════════════════════
// PARAMS
// ═══════════════════════════════════════════════════════════════
function updP() {
    const g = $('pG'),
        fs = S.sr,
        ny = fs / 2;
    if (!S.sig) {
        g.innerHTML = '';
        return;
    }
    const hp = Math.max(1, Math.floor(S.nfft * (1 - S.olP / 100)));
    if (S.tab === 'fs') {
        g.innerHTML =
            pk('fs', fs.toLocaleString(), 'Hz', '', 'pfs') +
            pk(
                'Window',
                S.fsW,
                'samples',
                ((S.fsW / fs) * 1e3).toFixed(1) + ' ms',
                'fsW',
            ) +
            pk(
                'f₁',
                (fs / S.fsW).toFixed(1),
                'Hz',
                'fundamental',
                'fsW',
            ) +
            pk(
                'Harmonics',
                S.fsH,
                '',
                'of ' + Math.floor(S.fsW / 2),
                'fsH',
            );
    } else if (S.tab === 'fft') {
        let n = 1;
        while (n < S.sig.length * S.zpR) n <<= 1;
        g.innerHTML =
            pk('fs', fs.toLocaleString(), 'Hz', '', 'pfs') +
            pk('Nyquist', ny.toLocaleString(), 'Hz', '', 'pny') +
            pk('N', S.sig.length.toLocaleString(), '', '', 'pN') +
            pk('Δf', (fs / n).toFixed(3), 'Hz', '', 'pdf') +
            pk(
                'Bins',
                (n / 2 + 1).toLocaleString(),
                '',
                '',
                'pbn',
            ) +
            pk(
                'Band',
                S.bLo + '-' + Math.min(S.bHi, Math.floor(ny)),
                'Hz',
                'extracted',
                'bandSel',
            );
    } else {
        let fS = 1;
        while (fS < S.nfft * S.zpR) fS <<= 1;
        const nb = fS / 2 + 1,
            nf = Math.max(
                1,
                Math.floor((S.sig.length - S.nfft) / hp) + 1,
            );
        g.innerHTML =
            pk('fs', fs.toLocaleString(), 'Hz', '', 'pfs') +
            pk(
                'n_fft',
                S.nfft,
                '',
                '' + ((S.nfft / fs) * 1e3).toFixed(1) + ' ms',
                'nf',
            ) +
            pk('Hop', hp, '', '', 'ol') +
            pk('Δf', (fs / S.nfft).toFixed(1), 'Hz', '', 'pdf') +
            pk(
                'Δt',
                ((hp / fs) * 1e3).toFixed(1),
                'ms',
                '',
                'pdt',
            ) +
            pk('Shape', nb + '×' + nf, '', '', 'psh') +
            pk(
                'Band',
                S.bLo + '-' + Math.min(S.bHi, Math.floor(ny)),
                'Hz',
                '',
                'bandSel',
            );
    }
}
function pk(l, v, u, d, lk) {
    return `<div class="pc" onclick="L('${lk}')"><div class="pl">${l}</div><div class="pv">${v}<span class="pu">${u}</span></div><div class="pd">${d}</div></div>`;
}
