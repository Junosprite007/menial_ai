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
            if (d.error) { alert('Error: ' + d.error); return; }
            if (d.components && d.components.length > 0) {
                showDecompositionResults(d);
            } else {
                alert(d.message || 'Analysis complete');
            }
        })
        .catch(e => {
            btn.textContent = 'Analyze in Python';
            btn.disabled = false;
            alert('Server error \u2014 is server.py running?\n' + e.message);
        });
}

function showDecompositionResults(d) {
    // Remove any existing results panel
    const old = document.getElementById('decomp-results');
    if (old) old.remove();

    const panel = document.createElement('div');
    panel.id = 'decomp-results';
    panel.style.cssText = 'position:fixed;top:0;right:0;width:480px;height:100vh;overflow-y:auto;' +
        'background:#0a0a1a;border-left:1px solid #333;z-index:10000;padding:20px;font-family:monospace;color:#ccc';

    let html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">' +
        '<h2 style="margin:0;color:#a78bfa;font-size:16px">NMF Decomposition (' + d.num_components + ' components)</h2>' +
        '<button onclick="document.getElementById(\'decomp-results\').remove()" ' +
        'style="background:none;border:1px solid #555;color:#ccc;padding:4px 10px;cursor:pointer;border-radius:4px">Close</button></div>';

    if (d.spectrogram_url) {
        html += '<img src="' + d.spectrogram_url + '" style="width:100%;border-radius:6px;margin-bottom:16px" />';
    }

    d.components.forEach(function(c) {
        html += '<div style="border:1px solid #333;border-radius:6px;padding:12px;margin-bottom:12px">' +
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px">' +
            '<span style="color:' + c.color + ';font-weight:bold">Component ' + c.id + '</span>' +
            '<span style="color:#888">' + Math.round(c.freq_peak) + ' Hz</span></div>';
        if (c.spectrogram_url) {
            html += '<img src="' + c.spectrogram_url + '" style="width:100%;border-radius:4px;margin-bottom:8px" />';
        }
        if (c.audio_url) {
            html += '<audio controls src="' + c.audio_url + '" style="width:100%;height:32px"></audio>';
        }
        html += '</div>';
    });

    panel.innerHTML = html;
    document.body.appendChild(panel);
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
