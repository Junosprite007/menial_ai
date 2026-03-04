// ═══════════════════════════════════════════════════════════════
// EVENTS
// ═══════════════════════════════════════════════════════════════
function init() {
    $('fin').addEventListener('change', e => {
        if (e.target.files[0]) ldFile(e.target.files[0]);
    });
    const z = $('uz');
    z.addEventListener('dragover', e => {
        e.preventDefault();
        z.classList.add('dg');
    });
    z.addEventListener('dragleave', () => z.classList.remove('dg'));
    z.addEventListener('drop', e => {
        e.preventDefault();
        z.classList.remove('dg');
        if (e.dataTransfer.files[0])
            ldFile(e.dataTransfer.files[0]);
    });
    document.body.addEventListener('dragover', e =>
        e.preventDefault(),
    );
    document.body.addEventListener('drop', e => {
        e.preventDefault();
        if (e.dataTransfer.files[0])
            ldFile(e.dataTransfer.files[0]);
    });

    document.querySelectorAll('#pBtns .bt').forEach(b =>
        b.addEventListener('click', () => {
            ldP(b.dataset.p);
            L('pre');
        }),
    );

    document.querySelectorAll('.tab').forEach(t =>
        t.addEventListener('click', () => {
            S.tab = t.dataset.t;
            document
                .querySelectorAll('.tab')
                .forEach(x => x.classList.toggle('ac', x === t));
            const isFS = S.tab === 'fs';
            $('fsCtrls').style.display = isFS ? 'none' : '';
            $('fSerC').style.display = isFS ? '' : 'none';
            $('fsViz').style.display = isFS ? '' : 'none';
            $('stC').style.display = S.tab === 'stft' ? '' : 'none';
            $('bandSec').style.display = isFS ? 'none' : '';
            $('livS').style.display =
                S.playing && S.anOn && !isFS ? '' : 'none';
            sCK = '';
            recomputeExtraction();
            if (!S.playing) rAll();
            else rFr();
            updP();
            L(t.dataset.l);
        }),
    );

    document.querySelectorAll('#wBtns .bt').forEach(b =>
        b.addEventListener('click', () => {
            S.wt = b.dataset.w;
            document
                .querySelectorAll('#wBtns .bt')
                .forEach(x => x.classList.toggle('ac', x === b));
            sCK = '';
            if (!S.playing) rAll();
            L(b.dataset.l);
        }),
    );
    document.querySelectorAll('#nfB .bt').forEach(b =>
        b.addEventListener('click', () => {
            S.nfft = parseInt(b.dataset.n);
            document
                .querySelectorAll('#nfB .bt')
                .forEach(x => x.classList.toggle('ac', x === b));
            sCK = '';
            if (!S.playing) rAll();
            updP();
            L('nf');
        }),
    );
    document.querySelectorAll('#zpB .bt').forEach(b =>
        b.addEventListener('click', () => {
            S.zpR = parseInt(b.dataset.z);
            document
                .querySelectorAll('#zpB .bt')
                .forEach(x => x.classList.toggle('ac', x === b));
            sCK = '';
            if (!S.playing) rAll();
            L('zp');
        }),
    );

    $('mfS').addEventListener('input', e => {
        S.mf = +e.target.value;
        $('mfV').textContent = S.mf + ' Hz';
        sCK = '';
        if (!S.playing) rAll();
    });
    $('drS').addEventListener('input', e => {
        S.dynR = +e.target.value;
        $('drV').textContent = S.dynR + ' dB';
        sCK = '';
        if (!S.playing) rAll();
    });
    $('olS').addEventListener('input', e => {
        S.olP = +e.target.value;
        $('olV').textContent = S.olP + '%';
        sCK = '';
        if (!S.playing) rAll();
        updP();
    });

    // Band sliders (single-band mode)
    $('bandLo').addEventListener('input', e => {
        S.bLo = +e.target.value;
        if (S.bLo > S.bHi) {
            S.bHi = S.bLo;
            $('bandHi').value = S.bHi;
            $('bandHiV').textContent = S.bHi + ' Hz';
        }
        $('bandLoV').textContent = S.bLo + ' Hz';
        recomputeExtraction();
        if (!S.playing) rAll();
        updP();
    });
    $('bandHi').addEventListener('input', e => {
        S.bHi = +e.target.value;
        if (S.bHi < S.bLo) {
            S.bLo = S.bHi;
            $('bandLo').value = S.bLo;
            $('bandLoV').textContent = S.bLo + ' Hz';
        }
        $('bandHiV').textContent = S.bHi + ' Hz';
        recomputeExtraction();
        if (!S.playing) rAll();
        updP();
    });

    // Rolloff slider
    $('rolloffS').addEventListener('input', e => {
        S.rolloff = +e.target.value;
        $('rolloffV').textContent = S.rolloff;
        recomputeExtraction();
        if (!S.playing) rAll();
    });

    // Wet/dry slider
    $('wetDry').addEventListener('input', e => {
        S.wetDry = +e.target.value / 100;
        S.prevWetDry = S.wetDry;
        $('wetDryV').textContent = Math.round(S.wetDry * 100) + '%';
    });

    // Mode selector
    document
        .querySelectorAll('#modeBtns .bt')
        .forEach(b =>
            b.addEventListener('click', () =>
                switchMode(b.dataset.m),
            ),
        );

    // Order selector (combined mode)
    document.querySelectorAll('#orderBtns .bt').forEach(b =>
        b.addEventListener('click', () => {
            S.combOrder = b.dataset.o;
            document
                .querySelectorAll('#orderBtns .bt')
                .forEach(x => x.classList.toggle('ac', x === b));
            recomputeExtraction();
            if (!S.playing) rAll();
        }),
    );

    // Spectral subtraction sliders (standalone panel)
    function setupNoiseSliders(
        startId,
        endId,
        startVId,
        endVId,
        overSubId,
        overSubVId,
        specFloorId,
        specFloorVId,
    ) {
        const startEl = $(startId),
            endEl = $(endId);
        if (startEl)
            startEl.addEventListener('input', e => {
                S.noiseStart = +e.target.value / 1000;
                const dur = S.sig ? S.sig.length / S.sr : 1;
                $(startVId).textContent =
                    (S.noiseStart * dur).toFixed(2) + 's';
                // sync other panel
                if (
                    startId === 'noiseStartS' &&
                    $('cNoiseStartS')
                ) {
                    $('cNoiseStartS').value = e.target.value;
                    $('cNoiseStartV').textContent =
                        $(startVId).textContent;
                }
                if (
                    startId === 'cNoiseStartS' &&
                    $('noiseStartS')
                ) {
                    $('noiseStartS').value = e.target.value;
                    $('noiseStartV').textContent =
                        $(startVId).textContent;
                }
                if (!S.playing) rAll();
            });
        if (endEl)
            endEl.addEventListener('input', e => {
                S.noiseEnd = +e.target.value / 1000;
                const dur = S.sig ? S.sig.length / S.sr : 1;
                $(endVId).textContent =
                    (S.noiseEnd * dur).toFixed(2) + 's';
                if (endId === 'noiseEndS' && $('cNoiseEndS')) {
                    $('cNoiseEndS').value = e.target.value;
                    $('cNoiseEndV').textContent =
                        $(endVId).textContent;
                }
                if (endId === 'cNoiseEndS' && $('noiseEndS')) {
                    $('noiseEndS').value = e.target.value;
                    $('noiseEndV').textContent =
                        $(endVId).textContent;
                }
                if (!S.playing) rAll();
            });
        if ($(overSubId))
            $(overSubId).addEventListener('input', e => {
                S.overSub = +e.target.value / 100;
                $(overSubVId).textContent = S.overSub.toFixed(1);
                // sync
                if (overSubId === 'overSubS' && $('cOverSubS')) {
                    $('cOverSubS').value = e.target.value;
                    $('cOverSubV').textContent =
                        S.overSub.toFixed(1);
                }
                if (overSubId === 'cOverSubS' && $('overSubS')) {
                    $('overSubS').value = e.target.value;
                    $('overSubV').textContent =
                        S.overSub.toFixed(1);
                }
                recomputeExtraction();
                if (!S.playing) rAll();
            });
        if ($(specFloorId))
            $(specFloorId).addEventListener('input', e => {
                S.specFloor = +e.target.value / 1000;
                $(specFloorVId).textContent =
                    S.specFloor.toFixed(3);
                if (
                    specFloorId === 'specFloorS' &&
                    $('cSpecFloorS')
                ) {
                    $('cSpecFloorS').value = e.target.value;
                    $('cSpecFloorV').textContent =
                        S.specFloor.toFixed(3);
                }
                if (
                    specFloorId === 'cSpecFloorS' &&
                    $('specFloorS')
                ) {
                    $('specFloorS').value = e.target.value;
                    $('specFloorV').textContent =
                        S.specFloor.toFixed(3);
                }
                recomputeExtraction();
                if (!S.playing) rAll();
            });
    }
    setupNoiseSliders(
        'noiseStartS',
        'noiseEndS',
        'noiseStartV',
        'noiseEndV',
        'overSubS',
        'overSubV',
        'specFloorS',
        'specFloorV',
    );
    setupNoiseSliders(
        'cNoiseStartS',
        'cNoiseEndS',
        'cNoiseStartV',
        'cNoiseEndV',
        'cOverSubS',
        'cOverSubV',
        'cSpecFloorS',
        'cSpecFloorV',
    );

    // Spectrogram click-drag for band selection
    const mcEl = $('mC');
    let dragState = null;
    mcEl.addEventListener('mousedown', e => {
        if (S.tab !== 'stft') return;
        const rect = mcEl.getBoundingClientRect();
        const stftP = { l: 42, r: 8, t: 8, b: 26 };
        const stftPH = rect.height - stftP.t - stftP.b;
        const y = e.clientY - rect.top;
        const stftDm = Math.min(S.mf, S.sr / 2);
        const freq = ((stftP.t + stftPH - y) / stftPH) * stftDm;
        dragState = {
            startFreq: Math.max(0, freq),
            shiftKey: e.shiftKey,
            startY: y,
        };
        $('mCov').classList.add('active');
        $('mCW').classList.add('crosshair');
    });
    document.addEventListener('mousemove', e => {
        if (!dragState) return;
        const ov = $('mCov');
        const rect = mcEl.getBoundingClientRect();
        const dp = devicePixelRatio || 1;
        ov.width = rect.width * dp;
        ov.height = rect.height * dp;
        ov.style.width = rect.width + 'px';
        ov.style.height = rect.height + 'px';
        const ctx = ov.getContext('2d');
        ctx.scale(dp, dp);
        ctx.clearRect(0, 0, rect.width, rect.height);
        const stftP = { l: 42, r: 8, t: 8, b: 26 };
        const stftPW = rect.width - stftP.l - stftP.r;
        const stftPH = rect.height - stftP.t - stftP.b;
        const y = e.clientY - rect.top;
        const y0 = Math.min(dragState.startY, y),
            y1 = Math.max(dragState.startY, y);
        ctx.fillStyle = 'rgba(34,211,238,.15)';
        ctx.fillRect(
            stftP.l,
            Math.max(y0, stftP.t),
            stftPW,
            Math.min(y1, stftP.t + stftPH) - Math.max(y0, stftP.t),
        );
        ctx.strokeStyle = 'rgba(34,211,238,.6)';
        ctx.lineWidth = 1;
        ctx.strokeRect(
            stftP.l,
            Math.max(y0, stftP.t),
            stftPW,
            Math.min(y1, stftP.t + stftPH) - Math.max(y0, stftP.t),
        );
    });
    document.addEventListener('mouseup', e => {
        if (!dragState) return;
        $('mCov').classList.remove('active');
        $('mCW').classList.remove('crosshair');
        const ov = $('mCov');
        const ctx = ov.getContext('2d');
        ctx.clearRect(0, 0, ov.width, ov.height);
        const rect = mcEl.getBoundingClientRect();
        const stftP = { l: 42, r: 8, t: 8, b: 26 };
        const stftPH = rect.height - stftP.t - stftP.b;
        const y = e.clientY - rect.top;
        const stftDm = Math.min(S.mf, S.sr / 2);
        const endFreq = Math.max(
            0,
            ((stftP.t + stftPH - y) / stftPH) * stftDm,
        );
        const lo = Math.max(
            0,
            Math.min(dragState.startFreq, endFreq),
        );
        const hi = Math.min(
            stftDm,
            Math.max(dragState.startFreq, endFreq),
        );
        if (Math.abs(hi - lo) < 20) {
            dragState = null;
            return;
        }
        if (S.isoMode === 'single') {
            S.bLo = Math.round(lo);
            S.bHi = Math.round(hi);
            $('bandLo').value = S.bLo;
            $('bandLoV').textContent = S.bLo + ' Hz';
            $('bandHi').value = S.bHi;
            $('bandHiV').textContent = S.bHi + ' Hz';
        } else if (
            S.isoMode === 'multi' ||
            S.isoMode === 'combined'
        ) {
            if (dragState.shiftKey) {
                addBand(lo, hi);
            } else {
                S.bands = [
                    {
                        lo: Math.round(lo),
                        hi: Math.round(hi),
                        rolloff: 4,
                    },
                ];
                renderBands();
            }
        }
        dragState = null;
        recomputeExtraction();
        if (!S.playing) rAll();
        updP();
    });

    // Waveform region selection (drag to select)
    const wCEl = $('wC');
    let wDrag = null;
    $('wCW').addEventListener('mousedown', e => {
        if (e.target.tagName === 'INPUT') return;
        const rect = wCEl.getBoundingClientRect();
        const frac = Math.max(
            0,
            Math.min(1, (e.clientX - rect.left) / rect.width),
        );
        wDrag = { startFrac: frac };
        e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
        if (!wDrag) return;
        const rect = wCEl.getBoundingClientRect();
        const frac = Math.max(
            0,
            Math.min(1, (e.clientX - rect.left) / rect.width),
        );
        const lo = Math.min(wDrag.startFrac, frac);
        const hi = Math.max(wDrag.startFrac, frac);
        if (hi - lo > 0.005) {
            S.selStart = lo;
            S.selEnd = hi;
            S.hasSelection = true;
            if (!S.playing) rAll();
        }
    });
    document.addEventListener('mouseup', e => {
        if (!wDrag) return;
        const rect = wCEl.getBoundingClientRect();
        const frac = Math.max(
            0,
            Math.min(1, (e.clientX - rect.left) / rect.width),
        );
        const lo = Math.min(wDrag.startFrac, frac);
        const hi = Math.max(wDrag.startFrac, frac);
        if (hi - lo > 0.005) {
            S.selStart = lo;
            S.selEnd = hi;
            S.hasSelection = true;
        } else {
            // Click without drag = seek to position
            seekTo(frac);
        }
        wDrag = null;
        if (!S.playing) rAll();
    });
    // Double-click waveform to clear selection
    $('wCW').addEventListener('dblclick', e => {
        if (e.target.tagName === 'INPUT') return;
        S.hasSelection = false;
        S.selStart = 0;
        S.selEnd = 1;
        if (!S.playing) rAll();
        L('wf');
    });

    // Editable number values
    makeEditable($('bandLoV'), {
        min: 0,
        max: S.sr / 2,
        step: 10,
        suffix: ' Hz',
        onChange: v => {
            S.bLo = v;
            $('bandLo').value = v;
            if (S.bLo > S.bHi) {
                S.bHi = S.bLo;
                $('bandHi').value = S.bHi;
                $('bandHiV').textContent = S.bHi + ' Hz';
            }
            recomputeExtraction();
            if (!S.playing) rAll();
            updP();
        },
    });
    makeEditable($('bandHiV'), {
        min: 0,
        max: S.sr / 2,
        step: 10,
        suffix: ' Hz',
        onChange: v => {
            S.bHi = v;
            $('bandHi').value = v;
            if (S.bHi < S.bLo) {
                S.bLo = S.bHi;
                $('bandLo').value = S.bLo;
                $('bandLoV').textContent = S.bLo + ' Hz';
            }
            recomputeExtraction();
            if (!S.playing) rAll();
            updP();
        },
    });
    makeEditable($('rolloffV'), {
        min: 1,
        max: 12,
        step: 1,
        suffix: '',
        onChange: v => {
            S.rolloff = v;
            $('rolloffS').value = v;
            recomputeExtraction();
            if (!S.playing) rAll();
        },
    });
    makeEditable($('mfV'), {
        min: 500,
        max: 22050,
        step: 250,
        suffix: ' Hz',
        onChange: v => {
            S.mf = v;
            $('mfS').value = v;
            sCK = '';
            if (!S.playing) rAll();
        },
    });
    makeEditable($('drV'), {
        min: 20,
        max: 120,
        step: 5,
        suffix: ' dB',
        onChange: v => {
            S.dynR = v;
            $('drS').value = v;
            sCK = '';
            if (!S.playing) rAll();
        },
    });
    makeEditable($('olV'), {
        min: 0,
        max: 90,
        step: 5,
        suffix: '%',
        onChange: v => {
            S.olP = v;
            $('olS').value = v;
            sCK = '';
            if (!S.playing) rAll();
            updP();
        },
    });
    makeEditable($('fsWV'), {
        min: 64,
        max: 8192,
        step: 64,
        suffix: '',
        onChange: v => {
            S.fsW = v;
            $('fsWS').value = v;
            $('fsHS').max = Math.floor(v / 2);
            if (S.fsH > Math.floor(v / 2)) {
                S.fsH = Math.floor(v / 2);
                $('fsHS').value = S.fsH;
                $('fsHV').textContent = S.fsH;
            }
            recomputeExtraction();
            if (!S.playing) rAll();
            updP();
        },
    });
    makeEditable($('fsHV'), {
        min: 1,
        max: 64,
        step: 1,
        suffix: '',
        onChange: v => {
            S.fsH = v;
            $('fsHS').value = v;
            recomputeExtraction();
            if (!S.playing) rAll();
            updP();
        },
    });
    makeEditable($('wetDryV'), {
        min: 0,
        max: 100,
        step: 1,
        suffix: '%',
        onChange: v => {
            S.wetDry = v / 100;
            S.prevWetDry = S.wetDry;
            $('wetDry').value = v;
        },
    });
    makeEditable($('overSubV'), {
        min: 0.5,
        max: 4.0,
        step: 0.1,
        suffix: '',
        onChange: v => {
            S.overSub = v;
            $('overSubS').value = Math.round(v * 100);
            if ($('cOverSubS')) {
                $('cOverSubS').value = Math.round(v * 100);
                $('cOverSubV').textContent = v.toFixed(1);
            }
            recomputeExtraction();
            if (!S.playing) rAll();
        },
    });
    makeEditable($('specFloorV'), {
        min: 0.001,
        max: 0.1,
        step: 0.001,
        suffix: '',
        fmt: v => v.toFixed(3),
        onChange: v => {
            S.specFloor = v;
            $('specFloorS').value = Math.round(v * 1000);
            if ($('cSpecFloorS')) {
                $('cSpecFloorS').value = Math.round(v * 1000);
                $('cSpecFloorV').textContent = v.toFixed(3);
            }
            recomputeExtraction();
            if (!S.playing) rAll();
        },
    });

    // FS sliders
    $('fsWS').addEventListener('input', e => {
        S.fsW = +e.target.value;
        $('fsWV').textContent = S.fsW;
        $('fsHS').max = Math.floor(S.fsW / 2);
        if (S.fsH > Math.floor(S.fsW / 2)) {
            S.fsH = Math.floor(S.fsW / 2);
            $('fsHS').value = S.fsH;
            $('fsHV').textContent = S.fsH;
        }
        recomputeExtraction();
        if (!S.playing) rAll();
        updP();
    });
    $('fsHS').addEventListener('input', e => {
        S.fsH = +e.target.value;
        $('fsHV').textContent = S.fsH;
        recomputeExtraction();
        if (!S.playing) rAll();
        updP();
    });
    $('fsOS').addEventListener('input', e => {
        S.fsOff = +e.target.value;
        $('fsOV').textContent = Math.floor(
            (S.fsOff / 1000) *
                Math.max(0, S.sig ? S.sig.length - S.fsW : 0),
        );
        recomputeExtraction();
        if (!S.playing) rAll();
    });

    const sk = $('seek');
    let sking = false;
    sk.addEventListener('mousedown', () => {
        sking = true;
    });
    sk.addEventListener('input', () => {
        if (sking) seekTo(+sk.value / 1e3);
    });
    sk.addEventListener('mouseup', () => {
        sking = false;
    });
    sk.addEventListener('change', () => {
        seekTo(+sk.value / 1e3);
        sking = false;
    });

    document.addEventListener('keydown', e => {
        if (
            e.target.tagName === 'INPUT' ||
            e.target.classList.contains('inline-edit')
        )
            return;
        if (e.code === 'Space') {
            e.preventDefault();
            togPlay();
        }
        if (e.code === 'Escape') stopA();
        if (e.code === 'KeyA') togAn();
        if (e.code === 'KeyE') togExtPlay();
    });
    window.addEventListener('resize', () => {
        sCK = '';
        if (!S.playing) rAll();
    });
}

// Populate educational panels
if ($('eduSingle')) $('eduSingle').innerHTML = EDU.single;
if ($('eduMulti')) $('eduMulti').innerHTML = EDU.multi;
if ($('eduSpectral')) $('eduSpectral').innerHTML = EDU.spectral;
if ($('eduCombined')) $('eduCombined').innerHTML = EDU.combined;

ldP('tones');
init();
renderBands();
