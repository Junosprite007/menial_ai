// ── MODE SWITCHING ──
function switchMode(mode) {
    S.isoMode = mode;
    document
        .querySelectorAll('#modeBtns .bt')
        .forEach(b =>
            b.classList.toggle('ac', b.dataset.m === mode),
        );
    $('panSingle').style.display = mode === 'single' ? '' : 'none';
    $('panMulti').style.display = mode === 'multi' ? '' : 'none';
    $('panSpectral').style.display =
        mode === 'spectral' ? '' : 'none';
    $('panCombined').style.display =
        mode === 'combined' ? '' : 'none';
    renderBands();
    recomputeExtraction();
    if (!S.playing) rAll();
}

// ── MULTI-BAND UI ──
function addBand(lo, hi) {
    const ny = S.sr / 2;
    S.bands.push({
        lo: lo !== undefined ? Math.round(lo) : 200,
        hi: hi !== undefined ? Math.round(hi) : 2000,
        rolloff: 4,
    });
    renderBands();
    recomputeExtraction();
    if (!S.playing) rAll();
}
function removeBand(idx) {
    S.bands.splice(idx, 1);
    renderBands();
    recomputeExtraction();
    if (!S.playing) rAll();
}
function renderBands() {
    const container =
        S.isoMode === 'combined' ? $('cBandList') : $('bandList');
    const other =
        S.isoMode === 'combined' ? $('bandList') : $('cBandList');
    if (other) other.innerHTML = '';
    if (!container) return;
    container.innerHTML = '';
    const ny = S.sr / 2;
    S.bands.forEach((b, idx) => {
        const col = HC[idx % HC.length];
        const row = document.createElement('div');
        row.className = 'band-row';
        row.innerHTML = `
            <div class="band-color" style="background:${col}"></div>
            <div class="band-ctrls">
                <div class="sr">
                    <span class="srl">Low</span>
                    <input type="range" min="0" max="${ny}" step="10" value="${b.lo}" data-idx="${idx}" data-f="lo" />
                    <span class="srv band-val" data-idx="${idx}" data-f="lo">${b.lo} Hz</span>
                </div>
                <div class="sr">
                    <span class="srl">High</span>
                    <input type="range" min="0" max="${ny}" step="10" value="${b.hi}" data-idx="${idx}" data-f="hi" />
                    <span class="srv band-val" data-idx="${idx}" data-f="hi">${b.hi} Hz</span>
                </div>
                <div class="sr">
                    <span class="srl">Rolloff</span>
                    <input type="range" min="1" max="12" step="1" value="${b.rolloff}" data-idx="${idx}" data-f="rolloff" />
                    <span class="srv band-val" data-idx="${idx}" data-f="rolloff">${b.rolloff}</span>
                </div>
            </div>
            <button class="band-rm" onclick="removeBand(${idx})">✕</button>
        `;
        row.querySelectorAll('input[type=range]').forEach(inp => {
            inp.addEventListener('input', e => {
                const i = +e.target.dataset.idx,
                    f = e.target.dataset.f;
                S.bands[i][f] = +e.target.value;
                const vEl = row.querySelector(
                    `.band-val[data-idx="${i}"][data-f="${f}"]`,
                );
                if (vEl)
                    vEl.textContent =
                        e.target.value +
                        (f === 'rolloff' ? '' : ' Hz');
                recomputeExtraction();
                if (!S.playing) rAll();
            });
        });
        container.appendChild(row);
    });
}

// ── EDITABLE NUMBER VALUES ──
function makeEditable(el, opts) {
    el.style.cursor = 'pointer';
    el.title = 'Double-click to type exact value';
    el.addEventListener('dblclick', function (e) {
        e.stopPropagation();
        const current = parseFloat(el.textContent);
        const input = document.createElement('input');
        input.type = 'text';
        input.value = isNaN(current) ? 0 : current;
        input.className = 'inline-edit';
        el.textContent = '';
        el.appendChild(input);
        input.focus();
        input.select();
        function commit() {
            let v = parseFloat(input.value);
            if (isNaN(v)) v = current;
            v = Math.max(opts.min, Math.min(opts.max, v));
            if (opts.step)
                v = Math.round(v / opts.step) * opts.step;
            el.textContent =
                (opts.fmt ? opts.fmt(v) : v) + (opts.suffix || '');
            opts.onChange(v);
        }
        input.addEventListener('keydown', ev => {
            if (ev.key === 'Enter') {
                ev.preventDefault();
                commit();
            }
            if (ev.key === 'Escape') {
                el.textContent =
                    (opts.fmt ? opts.fmt(current) : current) +
                    (opts.suffix || '');
            }
            ev.stopPropagation();
        });
        input.addEventListener('blur', commit);
    });
}

function togOverlay() {
    S.showOv = !S.showOv;
    $('ovTog').classList.toggle('on', S.showOv);
    if (!S.playing) rAll();
}
function togAn() {
    S.anOn = !S.anOn;
    $('anTog').classList.toggle('on', S.anOn);
    $('livS').style.display =
        S.playing && S.anOn && S.tab !== 'fs' ? '' : 'none';
    if (!S.playing) rAll();
}
