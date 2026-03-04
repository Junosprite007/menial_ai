// ═══════════════════════════════════════════════════════════════
// GENERATORS
// ═══════════════════════════════════════════════════════════════
function gTones(fs, d) {
    const N = Math.floor(fs * d),
        s = new Float64Array(N);
    for (let i = 0; i < N; i++) {
        const t = i / fs;
        s[i] =
            0.7 * Math.sin(2 * Math.PI * 440 * t) +
            0.5 * Math.sin(2 * Math.PI * 1e3 * t) +
            0.3 * Math.sin(2 * Math.PI * 2500 * t) +
            0.05 * (Math.random() * 2 - 1);
    }
    return s;
}
function gChirp(fs, d) {
    const N = Math.floor(fs * d),
        s = new Float64Array(N);
    for (let i = 0; i < N; i++) {
        const t = i / fs;
        s[i] =
            0.8 *
            Math.sin(
                2 * Math.PI * (100 * t + (0.5 * 3900 * t * t) / d),
            );
    }
    return s;
}
function gImp(fs, d) {
    const N = Math.floor(fs * d),
        s = new Float64Array(N),
        iv = Math.floor(fs * 0.35);
    for (let i = 0; i < N; i++) {
        const m = i % iv;
        if (m < Math.floor(fs * 0.012))
            s[i] = 0.9 * Math.exp(-m / (fs * 0.003));
    }
    return s;
}
function gChord(fs, d) {
    const f = [261.63, 329.63, 392, 523.25, 659.25, 784],
        a = [0.5, 0.4, 0.45, 0.25, 0.2, 0.22],
        N = Math.floor(fs * d),
        s = new Float64Array(N);
    for (let i = 0; i < N; i++) {
        const t = i / fs;
        for (let j = 0; j < f.length; j++)
            s[i] += a[j] * Math.sin(2 * Math.PI * f[j] * t);
        s[i] += 0.03 * (Math.random() * 2 - 1);
    }
    return s;
}
function gSq(fs, d) {
    const N = Math.floor(fs * d),
        s = new Float64Array(N);
    for (let i = 0; i < N; i++)
        s[i] =
            Math.sin((2 * Math.PI * 200 * i) / fs) >= 0
                ? 0.8
                : -0.8;
    return s;
}
