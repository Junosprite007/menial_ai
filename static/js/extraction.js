// ═══════════════════════════════════════════════════════════════
// EXTRACTION
// ═══════════════════════════════════════════════════════════════
function recomputeExtraction() {
    if (!S.sig) return;
    if (S.tab === 'fs') {
        const off = Math.floor(
            (S.fsOff / 1000) * Math.max(0, S.sig.length - S.fsW),
        );
        const { harmonics, chunk } = compFS(S.sig, off, S.fsW);
        const recon = reconstructFS(harmonics, S.fsW, S.fsH);
        const out = new Float64Array(S.sig.length);
        for (let i = 0; i < out.length; i++)
            out[i] = recon[i % S.fsW];
        S.fsSig = out;
        S.fsBuf = null;
    } else {
        switch (S.isoMode) {
            case 'single':
                S.extSig = smoothBandpass(
                    S.sig,
                    S.sr,
                    S.bLo,
                    S.bHi,
                    S.rolloff,
                );
                break;
            case 'multi':
                S.extSig = multiBandFilter(S.sig, S.sr, S.bands);
                break;
            case 'spectral':
                if (S.noiseProfile) {
                    S.extSig = spectralSubtract(
                        S.sig,
                        S.sr,
                        S.noiseProfile,
                        S.noiseFFTSize,
                        S.overSub,
                        S.specFloor,
                    );
                } else {
                    S.extSig = S.sig.slice();
                }
                break;
            case 'combined': {
                let processed = S.sig;
                if (S.combOrder === 'spectral-first') {
                    if (S.noiseProfile)
                        processed = spectralSubtract(
                            processed,
                            S.sr,
                            S.noiseProfile,
                            S.noiseFFTSize,
                            S.overSub,
                            S.specFloor,
                        );
                    if (S.bands.length > 0)
                        processed = multiBandFilter(
                            processed,
                            S.sr,
                            S.bands,
                        );
                } else {
                    if (S.bands.length > 0)
                        processed = multiBandFilter(
                            processed,
                            S.sr,
                            S.bands,
                        );
                    if (S.noiseProfile)
                        processed = spectralSubtract(
                            processed,
                            S.sr,
                            S.noiseProfile,
                            S.noiseFFTSize,
                            S.overSub,
                            S.specFloor,
                        );
                }
                S.extSig = processed;
                break;
            }
        }
        S.extBuf = null;
    }
}

function getPlaySig() {
    const orig = S.sig;
    if (!orig) return null;
    let ext;
    if (S.tab === 'fs') {
        ext = S.fsSig || orig;
    } else {
        ext = S.extSig || orig;
    }
    // Blend original and extracted based on wet/dry
    let mixed;
    if (!S.playExt || S.wetDry <= 0.0) {
        mixed = orig;
    } else if (S.wetDry >= 1.0) {
        mixed = ext;
    } else {
        mixed = new Float64Array(orig.length);
        const w = S.wetDry;
        for (let i = 0; i < orig.length; i++)
            mixed[i] = (1 - w) * orig[i] + w * ext[i];
    }
    return mixed;
}
