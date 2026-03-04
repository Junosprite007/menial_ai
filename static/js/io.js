// ═══════════════════════════════════════════════════════════════
// LOADING
// ═══════════════════════════════════════════════════════════════
function ldP(p) {
    stopA();
    S.preset = p;
    const fs = 44100,
        d = 2;
    S.sr = fs;
    switch (p) {
        case 'tones':
            S.sig = gTones(fs, d);
            S.lbl = '440+1k+2.5k Hz';
            break;
        case 'chirp':
            S.sig = gChirp(fs, d);
            S.lbl = 'Chirp';
            break;
        case 'impulses':
            S.sig = gImp(fs, d);
            S.lbl = 'Impulses';
            break;
        case 'chord':
            S.sig = gChord(fs, d);
            S.lbl = 'C Major';
            break;
        case 'square':
            S.sig = gSq(fs, d);
            S.lbl = 'Square 200Hz';
            break;
    }
    S.dur = d;
    S.pOff = 0;
    S.aBuf = null;
    sCK = '';
    document
        .querySelectorAll('#pBtns .bt')
        .forEach(b => b.classList.toggle('ac', b.dataset.p === p));
    $('uz').style.display = '';
    $('finfo').style.display = 'none';
    $('mfS').max = fs / 2;
    $('bandHi').max = fs / 2;
    $('bandLo').max = fs / 2;
    if (S.bHi > fs / 2) {
        S.bHi = fs / 2;
        $('bandHi').value = S.bHi;
        $('bandHiV').textContent = S.bHi + ' Hz';
    }
    updFSMax();
    recomputeExtraction();
    rAll();
}

function ldFile(file) {
    const rd = new FileReader();
    rd.onload = function (e) {
        try {
            stopA();
            const ctx = ensCtx();
            ctx.decodeAudioData(
                e.target.result,
                function (buf) {
                    let mono;
                    if (buf.numberOfChannels > 1) {
                        mono = new Float64Array(buf.length);
                        for (
                            let c = 0;
                            c < buf.numberOfChannels;
                            c++
                        ) {
                            const d = buf.getChannelData(c);
                            for (let i = 0; i < mono.length; i++)
                                mono[i] += d[i];
                        }
                        for (let i = 0; i < mono.length; i++)
                            mono[i] /= buf.numberOfChannels;
                    } else {
                        const r = buf.getChannelData(0);
                        mono = new Float64Array(r.length);
                        for (let i = 0; i < r.length; i++)
                            mono[i] = r[i];
                    }
                    const mx = buf.sampleRate * 30;
                    S.sig =
                        mono.length > mx ? mono.slice(0, mx) : mono;
                    S.sr = buf.sampleRate;
                    S.lbl = file.name;
                    S.rawFile = file;
                    S.dur = S.sig.length / S.sr;
                    S.pOff = 0;
                    S.aBuf = null;
                    sCK = '';
                    $('uz').style.display = 'none';
                    $('finfo').style.display = 'flex';
                    $('fname').textContent = file.name;
                    $('fmeta').textContent =
                        `${buf.sampleRate} Hz · ${buf.numberOfChannels}ch · ${(buf.length / buf.sampleRate).toFixed(2)}s`;
                    document
                        .querySelectorAll('#pBtns .bt')
                        .forEach(b => b.classList.remove('ac'));
                    const ny = buf.sampleRate / 2;
                    $('mfS').max = ny;
                    $('bandHi').max = ny;
                    $('bandLo').max = ny;
                    if (S.mf > ny) {
                        S.mf = Math.min(8e3, ny);
                        $('mfS').value = S.mf;
                        $('mfV').textContent = S.mf + ' Hz';
                    }
                    if (S.bHi > ny) {
                        S.bHi = ny;
                        $('bandHi').value = S.bHi;
                        $('bandHiV').textContent = S.bHi + ' Hz';
                    }
                    updFSMax();
                    recomputeExtraction();
                    rAll();
                },
                function (err) {
                    alert('Decode error: ' + err.message);
                },
            );
        } catch (err) {
            alert(err.message);
        }
    };
    rd.readAsArrayBuffer(file);
}

function updFSMax() {
    if (S.sig) {
        const mx = Math.min(S.sig.length, 8192);
        $('fsWS').max = mx;
        if (S.fsW > mx) {
            S.fsW = mx;
            $('fsWS').value = mx;
            $('fsWV').textContent = mx;
        }
        $('fsHS').max = Math.floor(S.fsW / 2);
    }
}

// ═══════════════════════════════════════════════════════════════
// PYTHON BRIDGE — send audio to Flask server for isolator
// ═══════════════════════════════════════════════════════════════

function sendToPython() {
    if (!S.rawFile && !S.sig) {
        alert('Load an audio file first');
        return;
    }
    const btn = $('pyBtn');
    btn.textContent = 'Sending\u2026';
    btn.disabled = true;

    const fd = new FormData();
    if (S.rawFile) {
        fd.append('audio', S.rawFile);
    } else {
        fd.append('audio', encodeWAV(S.sig, S.sr), 'synthetic.wav');
    }
    fd.append('nfft', S.nfft);
    fd.append('overlap', S.olP);
    fd.append('window', S.wt);
    fd.append('maxFreq', S.mf);

    fetch('/analyze', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(d => {
            btn.textContent = 'Analyze in Python';
            btn.disabled = false;
            if (d.error) alert('Error: ' + d.error);
        })
        .catch(e => {
            btn.textContent = 'Analyze in Python';
            btn.disabled = false;
            alert('Server error \u2014 is server.py running?\n' + e.message);
        });
}

function encodeWAV(samples, sampleRate) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, 'RIFF');
    v.setUint32(4, 36 + samples.length * 2, true);
    ws(8, 'WAVE');
    ws(12, 'fmt ');
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, 1, true);
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    ws(36, 'data');
    v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
}
