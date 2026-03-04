// ═══════════════════════════════════════════════════════════════
// AUDIO
// ═══════════════════════════════════════════════════════════════
function ensCtx() {
    if (!S.aCtx)
        S.aCtx = new (
            window.AudioContext || window.webkitAudioContext
        )();
    if (S.aCtx.state === 'suspended') S.aCtx.resume();
    return S.aCtx;
}
function buildBuf(sig) {
    const c = ensCtx(),
        b = c.createBuffer(1, sig.length, S.sr),
        ch = b.getChannelData(0);
    for (let i = 0; i < sig.length; i++)
        ch[i] = Math.max(-1, Math.min(1, sig[i]));
    return b;
}

function togPlay() {
    S.playing ? pauseA() : startA();
}
function startA() {
    if (!S.sig || !S.sig.length) return;
    ensCtx();
    const ps = getPlaySig();
    if (!ps) return;
    const buf = buildBuf(ps);
    // Duration is always from full signal
    S.dur = S.sig.length / S.sr;
    // Determine play region
    const regStart = S.hasSelection ? S.selStart * S.dur : 0;
    const regEnd = S.hasSelection ? S.selEnd * S.dur : S.dur;
    // If playhead is outside region or at end, reset to region start
    if (S.pOff < regStart || S.pOff >= regEnd - 0.01)
        S.pOff = regStart;
    const s = S.aCtx.createBufferSource();
    s.buffer = buf;
    s.connect(S.aCtx.destination);
    const playDuration = regEnd - S.pOff;
    s.onended = () => {
        if (S.playing) {
            S.playing = false;
            S.pOff = S.hasSelection ? regStart : S.dur;
            updTr();
            stopAL();
            if (!S.playing) rAll();
        }
    };
    s.start(0, S.pOff, playDuration);
    S.sN = s;
    S.pSt = S.aCtx.currentTime;
    S.playing = true;
    updTr();
    startAL();
}
function pauseA() {
    if (!S.playing) return;
    S.pOff = curT();
    if (S.sN) {
        try {
            S.sN.stop();
        } catch (e) {}
        S.sN = null;
    }
    S.playing = false;
    updTr();
    stopAL();
    rFr();
}
function stopA() {
    if (S.sN) {
        try {
            S.sN.stop();
        } catch (e) {}
        S.sN = null;
    }
    S.playing = false;
    S.pOff = S.hasSelection ? S.selStart * S.dur : 0;
    updTr();
    stopAL();
    rAll();
}
function curT() {
    if (!S.playing) return S.pOff;
    return Math.min(S.pOff + S.aCtx.currentTime - S.pSt, S.dur);
}
function seekTo(f) {
    const w = S.playing;
    if (w) {
        if (S.sN) {
            try {
                S.sN.stop();
            } catch (e) {}
            S.sN = null;
        }
        S.playing = false;
    }
    S.pOff = f * S.dur;
    w ? startA() : (updTr(), rFr());
}
function fmtT(s) {
    const m = Math.floor(s / 60),
        sc = s - m * 60;
    return m + ':' + sc.toFixed(1).padStart(4, '0');
}
function updTr() {
    const b = $('playB');
    b.textContent = S.playing ? '⏸' : '▶';
    b.classList.toggle('pl', S.playing && !S.playExt);
    b.classList.toggle('ext', S.playing && S.playExt);
    $('livB').style.display = S.playing ? '' : 'none';
    $('livS').style.display =
        S.playing && S.anOn && S.tab !== 'fs' ? '' : 'none';
}

function togExtPlay() {
    const was = S.playing;
    if (was) pauseA();
    S.playExt = !S.playExt;
    $('extPlayB').textContent = S.playExt
        ? 'EXTRACTED'
        : 'ORIGINAL';
    $('extPlayB').classList.toggle('ext', S.playExt);
    if (was) startA();
}
