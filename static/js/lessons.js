const $ = id => document.getElementById(id);

// ═══════════════════════════════════════════════════════════════
// LESSONS
// ═══════════════════════════════════════════════════════════════
const LS = {
    wf: {
        t: 'Waveform',
        b: '<h3>Waveform</h3><p>Amplitude vs time. When the <span style="color:#22d3ee">cyan overlay</span> is on, the extracted frequency band is drawn on top of the original (blue) waveform. This lets you visually see which parts of the signal fall within your selected band.</p><p><strong>Region selection:</strong> Drag horizontally to select a time region (green edges). Playback will loop within this region. Double-click to clear. Click to seek.</p><div class="kp">The audio you hear during "ORIGINAL" mode is unchanged. In "EXTRACTED" mode, you hear only the cyan portion — everything outside your band has been removed via inverse FFT.</div>',
    },
    fftT: {
        t: 'FFT',
        b: '<h3>FFT</h3><p>Decomposes the signal into frequency bins. The <span style="color:#22d3ee">cyan highlighted region</span> on the spectrum shows your selected extraction band. Everything inside this band can be isolated and played back.</p><div class="fm">X[k] = Σ x[n] · e^(-j·2π·k·n/N)</div>',
    },
    stftT: {
        t: 'STFT',
        b: '<h3>STFT</h3><p>Spectrogram view. The extraction band appears as a <span style="color:#22d3ee">cyan horizontal band</span> overlaid on the spectrogram, showing which frequencies are being extracted at every time step.</p>',
    },
    fsT: {
        t: 'Fourier Series',
        b: '<h3>Fourier Series</h3><p>Decomposes a periodic window into individual sine waves. Each harmonic is shown separately. The reconstruction (green) converges on the original (white) as you add harmonics.</p><div class="kp">In "EXTRACTED" playback mode with the Fourier Series tab, you hear the reconstruction from the selected number of harmonics — letting you hear what each harmonic contributes.</div>',
    },
    bandSel: {
        t: 'Frequency Band Extraction',
        b: '<h3>Band Extraction</h3><p>This is the core new feature. Set a low and high frequency cutoff to define a <em>bandpass filter</em>. The tool computes the FFT of your entire signal, zeros out all bins outside this range, then inverse-FFTs back to get the filtered audio.</p><p>Toggle the playback to <strong>EXTRACTED</strong> to hear only the frequencies in your band. The <span style="color:#22d3ee">cyan overlay</span> on the waveform shows this filtered signal in real time.</p><div class="kp">Try isolating 400-500 Hz on the three-tone preset — you\'ll hear only the 440 Hz component. Or on a real recording, isolate 2000-4000 Hz to hear just the high-frequency content like sibilance, hissing, or the snap of percussive sounds.</div>',
    },
    extToggle: {
        t: 'Extraction Overlay',
        b: '<h3>Overlay Toggle</h3><p>When on, the extracted (bandpass-filtered) signal is drawn in <span style="color:#22d3ee">cyan</span> over the original blue waveform. This visual overlay lets you see exactly which parts of the audio fall within your selected frequency band.</p><p>Loud regions of cyan = lots of energy in your band. Regions where cyan is flat but blue is active = energy outside your band.</p>',
    },
    anTog: {
        t: 'Analysis Toggle',
        b: '<h3>Analysis On/Off</h3><p>Toggles the FFT/STFT visualization. Audio playback and extraction still work with analysis off.</p>',
    },
    pb: {
        t: 'Playback',
        b: '<h3>Playback & Extraction</h3><p>The "ORIGINAL / EXTRACTED" button switches what you <em>hear</em>:</p><p><strong>ORIGINAL:</strong> The unmodified audio signal.</p><p><strong>EXTRACTED:</strong> Only the frequencies within your selected band (bandpass filtered via inverse FFT). In Fourier Series mode, you hear the reconstruction from the selected harmonics.</p><div class="kp">This is where analysis becomes tangible. You\'re not just seeing frequency content — you\'re hearing it isolated. This is conceptually identical to what NMF source separation does in your project: decompose a signal into components, then reconstruct only the ones you want.</div>',
    },
    nf: {
        t: 'n_fft',
        b: '<h3>n_fft (FFT Window Size)</h3><p>The fundamental time-frequency tradeoff parameter. Controls how many samples are analyzed in each STFT frame.</p><div class="fm">Δf = fs / n_fft &nbsp;&nbsp;|&nbsp;&nbsp; Δt = hop / fs</div><p><strong>256:</strong> ~5.8ms windows at 44.1kHz. Excellent time resolution, poor frequency resolution (172 Hz bins). Good for transient sounds like drum hits.</p><p><strong>1024:</strong> ~23ms windows. Balanced time/frequency tradeoff. Good default.</p><p><strong>4096:</strong> ~93ms windows. Excellent frequency resolution (10.8 Hz bins), but time events are smeared. Good for sustained tones, chords, machinery hum.</p><div class="kp">The Heisenberg-Gabor uncertainty principle: you cannot have both perfect time and frequency resolution simultaneously. Δf × Δt ≥ 1/(4π). This is a fundamental limit, not a software limitation.</div>',
    },
    ol: {
        t: 'Overlap',
        b: '<h3>Overlap Percentage</h3><p>How much consecutive STFT frames overlap in time.</p><div class="fm">hop = n_fft × (1 - overlap%)</div><p><strong>75% (default):</strong> hop = n_fft/4. Standard for Hann window — ensures perfect reconstruction and smooth time axis.</p><p><strong>50%:</strong> hop = n_fft/2. Fewer frames, coarser time resolution, but faster computation.</p><p><strong>90%:</strong> Very smooth spectrogram with many frames. Useful for detailed time analysis but computationally expensive.</p><div class="kp">With a Hann window, 75% overlap provides "constant overlap-add" (COLA) — meaning if you sum all the windowed frames, you recover the original signal perfectly. This is essential for proper STFT analysis.</div>',
    },
    wfn: {
        t: 'Window Function',
        b: '<h3>Window Functions</h3><p>Tapers signal edges to prevent <strong>spectral leakage</strong>. The FFT assumes the signal is periodic — without windowing, discontinuities at frame edges cause energy to "leak" into neighboring frequency bins.</p><div class="fm">windowed_signal[n] = signal[n] × window[n]</div><p>The window multiplies each sample, tapering smoothly to zero at the edges. This reduces leakage at the cost of slightly widening frequency peaks (the "main lobe").</p><div class="kp">The tradeoff: narrower main lobe = better frequency resolution but more leakage. Wider main lobe = less leakage but blurred frequency peaks. Hann is the best general-purpose choice.</div>',
    },
    wr: {
        t: 'Rectangular',
        b: '<h3>Rectangular Window</h3><p>No taper at all — every sample weighted equally. Equivalent to no windowing.</p><p><strong>Main lobe width:</strong> Narrowest (best resolution).</p><p><strong>Sidelobe level:</strong> -13 dB (worst leakage).</p><div class="kp">Only useful when the signal is guaranteed to be exactly periodic within the frame (rare in practice). For real-world audio, spectral leakage makes rectangular almost useless.</div>',
    },
    wh: {
        t: 'Hann',
        b: '<h3>Hann Window</h3><p>The go-to window for general audio analysis. Named after Julius von Hann (sometimes incorrectly called "Hanning").</p><div class="fm">w[n] = 0.5 × (1 - cos(2πn / (N-1)))</div><p><strong>Sidelobe level:</strong> -31 dB with fast rolloff.</p><p><strong>Main lobe:</strong> Moderate width.</p><div class="kp">Use Hann unless you have a specific reason not to. It provides the best balance of leakage suppression and frequency resolution for most audio analysis tasks.</div>',
    },
    whm: {
        t: 'Hamming',
        b: '<h3>Hamming Window</h3><p>Similar to Hann but with a small pedestal — the window doesn\'t quite reach zero at the edges.</p><div class="fm">w[n] = 0.54 - 0.46 × cos(2πn / (N-1))</div><p><strong>Sidelobe level:</strong> -42 dB (better first sidelobe than Hann).</p><p><strong>Rolloff:</strong> Slower than Hann for distant sidelobes.</p><div class="kp">Traditional default in speech processing (e.g., MFCCs, formant analysis). The non-zero edges can cause issues with overlap-add reconstruction.</div>',
    },
    wb: {
        t: 'Blackman',
        b: '<h3>Blackman Window</h3><p>Three-term cosine window with the strongest sidelobe suppression of the four options here.</p><div class="fm">w[n] = 0.42 - 0.5cos(2πn/(N-1)) + 0.08cos(4πn/(N-1))</div><p><strong>Sidelobe level:</strong> -58 dB (excellent suppression).</p><p><strong>Main lobe:</strong> Widest — frequency peaks are most blurred.</p><div class="kp">Use when you need very clean spectra with minimal leakage (e.g., identifying weak tones near strong ones). The wider main lobe means you may need a larger FFT to compensate for lost resolution.</div>',
    },
    zp: {
        t: 'Zero-Padding',
        b: '<h3>Zero-Padding</h3><p>Appends zeros to the signal before FFT, increasing the number of output bins.</p><p><strong>2×:</strong> Doubles bins — interpolates between existing values for smoother appearance.</p><p><strong>4×:</strong> Quadruples bins — very smooth spectral envelope.</p><div class="kp">Important: zero-padding does NOT improve real frequency resolution (Δf is still fs/N_original). It only interpolates between the bins you already have. Think of it as "upsampling" the frequency axis. Useful for visualization and peak detection, but don\'t mistake the smoother curve for actual new information.</div>',
    },
    dr: {
        t: 'Dynamic Range',
        b: '<h3>Dynamic Range</h3><p>Controls the dB floor for the display. Only affects visualization, not the actual data.</p><p><strong>40 dB:</strong> Shows signals down to 1/100th of peak amplitude. Good for strong, clean signals.</p><p><strong>60 dB:</strong> Shows down to 1/1000th. Good default.</p><p><strong>90-120 dB:</strong> Shows very quiet components. Useful for finding weak harmonics or noise floor.</p><div class="kp">If the spectrogram looks mostly bright, decrease the range. If it looks mostly dark, increase it. The goal is to see meaningful structure.</div>',
    },
    mf: {
        t: 'Max Frequency',
        b: '<h3>Max Display Frequency</h3><p>Crops the frequency axis display. Does not affect analysis — just zooms into the range you care about.</p><p>Most useful audio content lives below 8 kHz:</p><p><strong>Speech fundamentals:</strong> 80-300 Hz<br><strong>Speech formants:</strong> 300-3500 Hz<br><strong>Music fundamentals:</strong> 30-4000 Hz<br><strong>Household sounds:</strong> 50-8000 Hz<br><strong>Sibilance/cymbals:</strong> 6000-16000 Hz</p><div class="kp">For the construction audio, try 50-8000 Hz to focus on hammering, sawing, and motor sounds.</div>',
    },
    pre: {
        t: 'Presets',
        b: '<h3>Synthetic Signal Presets</h3><p>Clean, mathematically defined signals for learning and testing:</p><p><strong>440+1k+2.5k Hz:</strong> Three pure sine waves. Perfect for testing band extraction — isolate each tone by setting the band around its frequency. Try 400-500 Hz to hear only the 440 Hz component.</p><p><strong>Chirp:</strong> Frequency sweep from 100 to 4000 Hz. Shows as a diagonal line on the STFT spectrogram — a great visual demo of time-frequency analysis.</p><p><strong>Impulses:</strong> Periodic clicks with exponential decay. Broadband energy (all frequencies at once). Tests how your filter handles transients.</p><p><strong>C Major:</strong> Six-note chord with harmonics. Rich harmonic structure visible in Fourier Series mode.</p><p><strong>Square:</strong> Square wave at 200 Hz. Only odd harmonics (200, 600, 1000, 1400...) — the missing even harmonics are visible in the Fourier Series decomposition.</p>',
    },
    fi: {
        t: 'File Input',
        b: '<h3>Audio File Loading</h3><p>Supports WAV, MP3, OGG, FLAC, and M4A formats. The browser\'s Web Audio API handles decoding.</p><p><strong>Stereo → Mono:</strong> Multi-channel audio is automatically mixed down to mono (averaged across channels).</p><p><strong>Duration limit:</strong> 30 seconds maximum. Longer files are truncated. This keeps FFT computation fast enough for interactive use.</p><p><strong>Sample rate:</strong> Preserved from the original file. All frequency limits (Nyquist, max display freq, band cutoffs) automatically adjust to match.</p><div class="kp">For best results with feature isolation, use recordings where the target sound (e.g., hammering) is clearly present. The extraction tools work on frequency content — they can\'t separate sounds that occupy the exact same frequencies.</div>',
    },
    pfs: {
        t: 'Sample Rate',
        b: '<h3>Sample Rate (fs)</h3><p>The number of amplitude measurements taken per second. Determines the maximum frequency that can be represented.</p><div class="fm">fs samples/second</div><p><strong>44,100 Hz:</strong> CD quality. Captures frequencies up to 22,050 Hz — well above human hearing range (~20 kHz).</p><p><strong>48,000 Hz:</strong> Professional audio/video standard.</p><div class="kp">The sample rate is fixed by the audio file. All other frequency parameters (Nyquist, band cutoffs, frequency resolution) derive from it.</div>',
    },
    pny: {
        t: 'Nyquist',
        b: '<h3>Nyquist Frequency</h3><p>The highest frequency that can be represented at a given sample rate. Named after Harry Nyquist.</p><div class="fm">f_Nyquist = fs / 2</div><p>Frequencies above this limit would "alias" — appear as phantom lower frequencies. This is why the band cutoff sliders stop at Nyquist.</p><div class="kp">The Shannon-Nyquist theorem: to faithfully capture a frequency f, you must sample at ≥ 2f. At 44.1 kHz, the max is 22,050 Hz.</div>',
    },
    pN: {
        t: 'N (Total Samples)',
        b: '<h3>N — Total Samples</h3><p>The total number of amplitude values in the loaded signal.</p><div class="fm">Duration = N / fs</div><p>For a 2-second signal at 44.1 kHz: N = 88,200 samples. This is also the FFT size when computing the full-signal spectrum (non-STFT mode).</p>',
    },
    pdf: {
        t: 'Δf',
        b: '<h3>Δf — Frequency Resolution</h3><p>The spacing between adjacent frequency bins in the FFT output.</p><div class="fm">Δf = fs / N_fft</div><p>Smaller Δf = you can distinguish closer frequencies. But achieving smaller Δf requires a longer FFT window, which blurs time resolution.</p><p><strong>Example:</strong> At 44.1 kHz with n_fft=1024: Δf = 43.1 Hz. Two tones at 440 Hz and 460 Hz (20 Hz apart) would fall in the same bin and be indistinguishable. With n_fft=4096: Δf = 10.8 Hz — now they\'re in separate bins.</p>',
    },
    pdt: {
        t: 'Δt',
        b: '<h3>Δt — Time Resolution</h3><p>The time between consecutive STFT frames (spectrogram columns).</p><div class="fm">Δt = hop / fs = n_fft × (1 - overlap%) / fs</div><p>Smaller Δt = finer time resolution, more frames to compute. At n_fft=1024, 75% overlap, 44.1 kHz: Δt ≈ 5.8ms — each spectrogram column represents ~6ms of audio.</p>',
    },
    pbn: {
        t: 'Bins',
        b: '<h3>Frequency Bins</h3><p>The number of unique frequency values in the FFT output.</p><div class="fm">bins = N_fft / 2 + 1</div><p>Due to the symmetry of real-valued FFT, only the first half of the output is unique (plus the DC and Nyquist bins). With n_fft=1024: 513 bins from 0 Hz to Nyquist.</p>',
    },
    pfr: {
        t: 'Frames',
        b: '<h3>Time Frames</h3><p>The number of columns in the spectrogram — each one is a separate FFT of an overlapping window.</p><div class="fm">frames ≈ (N - n_fft) / hop + 1</div><p>More frames = finer time axis but slower computation. Controlled by n_fft and overlap percentage.</p>',
    },
    psh: {
        t: 'Shape',
        b: '<h3>Spectrogram Shape</h3><p>The dimensions of the spectrogram matrix: bins × frames. This is the 2D data structure that the STFT produces.</p><p>Each cell contains the magnitude (or dB value) at a specific frequency and time. The colormap maps these values to pixel colors.</p>',
    },
    fsW: {
        t: 'FS Window',
        b: '<h3>Fourier Series — Window Size</h3><p>The number of samples treated as one period of the signal. This determines the fundamental frequency:</p><div class="fm">f₁ = fs / window_size</div><p>All harmonics are integer multiples of f₁. Larger window = lower fundamental = more harmonics available for reconstruction.</p><div class="kp">The Fourier Series assumes the window is one complete period that repeats forever. This is different from the FFT, which analyzes the signal as a whole. The "tiled" view shows this periodic assumption — the window is repeated left and right.</div>',
    },
    fsH: {
        t: 'Harmonics',
        b: '<h3>Fourier Series — Number of Harmonics</h3><p>How many sine/cosine components to include in the reconstruction (k = 0 to k = harmonics).</p><p><strong>k=0:</strong> DC offset (average value).</p><p><strong>k=1:</strong> Fundamental frequency f₁.</p><p><strong>k=2,3,...:</strong> Overtones at 2f₁, 3f₁, etc.</p><p>Watch the green reconstruction converge on the white original as you add more harmonics. This is Fourier\'s insight: <em>any</em> periodic signal can be decomposed into a sum of sines and cosines.</p><div class="kp">For a square wave, you need many harmonics (odd only: 1, 3, 5, 7...) to approximate the sharp edges. The "Gibbs phenomenon" — the persistent overshoot near discontinuities — never fully goes away no matter how many harmonics you add.</div>',
    },
    fsO: {
        t: 'Offset',
        b: '<h3>Window Offset Position</h3><p>Where in the audio signal to grab the window for Fourier Series analysis. Drag to slide the analysis window through the signal.</p><p>The orange-highlighted region on the waveform shows the current window position. Different positions capture different local frequency content.</p><div class="kp">For a signal with changing frequency content (like the chirp preset), moving the offset changes which harmonics are detected. For a stationary signal (like the tones preset), all positions give similar results.</div>',
    },
    fsR: {
        t: 'Reconstruction',
        b: '<h3>Fourier Series Reconstruction</h3><p><strong>White line:</strong> Original signal within the selected window.</p><p><strong>Green line:</strong> Sum of the first k harmonics (reconstruction).</p><p>As you increase the number of harmonics, the green line converges toward the white line. With enough harmonics, the reconstruction becomes indistinguishable from the original.</p><p>In EXTRACTED playback mode with the Fourier Series tab active, you hear this green reconstruction tiled across the full signal duration.</p>',
    },
    fsHr: {
        t: 'Individual Harmonics',
        b: '<h3>Individual Harmonics</h3><p>Each row shows one harmonic component — a single sine wave at frequency k × f₁:</p><p><strong>k=0:</strong> DC component (flat line at the signal\'s average value).</p><p><strong>k=1:</strong> Fundamental — the lowest frequency, one complete cycle per window.</p><p><strong>k=2+:</strong> Overtones — integer multiples of the fundamental. Progressively higher frequencies, often progressively smaller amplitudes.</p><p>The amplitude and phase of each harmonic are extracted via FFT of the windowed signal. The color of each trace matches its position in the harmonic series.</p><div class="kp">In a rich musical tone (like the C Major chord), you\'ll see significant energy at many harmonics. In a pure sine wave, only k=1 has energy. The distribution of energy across harmonics is what gives sounds their unique "timbre" or character.</div>',
    },
    // v6 lessons
    isoMode: {
        t: 'Isolation Mode',
        b: '<h3>Isolation Modes</h3><p>Four different approaches to extracting sounds from audio:</p><p><strong>Single Band:</strong> Classic bandpass filter with smooth Butterworth rolloff. Best for isolating a single frequency range (e.g., just the bass, just the highs).</p><p><strong>Multi-Band:</strong> Define multiple frequency ranges simultaneously. Essential for sounds that span several frequency regions (e.g., a hammer strike has both low thud and high crack).</p><p><strong>Spectral Subtraction:</strong> Estimate background noise from a quiet section, then subtract it from the entire signal. Great for denoising.</p><p><strong>Combined:</strong> Chain spectral subtraction with multi-band filtering for maximum isolation power.</p><div class="kp">Each mode has its own collapsible educational panel — click "? Learn about this mode" to expand it.</div>',
    },
    wetDryCtrl: {
        t: 'Wet/Dry Mix',
        b: '<h3>Wet/Dry Mix</h3><p>Controls the blend between original ("dry") and processed ("wet") audio:</p><p><strong>0%:</strong> 100% original signal, no extraction applied.</p><p><strong>50%:</strong> Equal mix of original and extracted.</p><p><strong>100%:</strong> Fully extracted/processed signal only.</p><div class="fm">output = (1 - mix) × original + mix × extracted</div><p>The ORIGINAL/EXTRACTED toggle acts as a bypass — it instantly mutes/unmutes the extraction without moving the mix slider. This lets you A/B compare at any mix level.</p><div class="kp">Tip: Sweep the mix slider slowly while listening to hear exactly how the extraction changes the audio character. Values around 70-90% often sound more natural than 100% because they retain some original context.</div>',
    },
    rolloffCtrl: {
        t: 'Filter Rolloff',
        b: '<h3>Butterworth Rolloff Order</h3><p>Controls how sharply the filter transitions from passband (full signal) to stopband (silence). This is the "N" in the Butterworth formula:</p><div class="fm">|H(f)| = 1 / sqrt(1 + (f/fc)^(2N))</div><p>Each unit of order adds approximately <strong>20 dB/decade</strong> of rolloff steepness:</p><p><strong>Order 1:</strong> Very gentle — frequencies near the cutoff are only slightly attenuated. Good for subtle shaping.</p><p><strong>Order 4:</strong> Standard — a good balance between sharpness and smoothness.</p><p><strong>Order 8-12:</strong> Very sharp — approaches a brick-wall filter. Can cause "ringing" (oscillation near the cutoff) on transient sounds like drum hits.</p><div class="kp">The amber gain curve on the FFT plot shows the exact filter shape. Watch how it changes as you adjust the order.</div>',
    },
    noiseRegion: {
        t: 'Noise Region',
        b: '<h3>Noise Region Selection</h3><p>Spectral subtraction needs a reference: what does "just noise" sound like? Select a portion of the audio that contains <em>only</em> background noise (no target sounds).</p><p><strong>Noise start/end:</strong> Define the time range. The orange-highlighted region on the waveform shows your selection.</p><p><strong>Capture Noise Profile:</strong> Computes the FFT of the selected region and stores its magnitude spectrum as the noise template.</p><div class="kp">Tips for good noise profiles:<br>• Choose at least 0.2 seconds of noise<br>• Pick a section with consistent, representative noise<br>• Avoid sections with any target sounds (even faint ones)<br>• If results are poor, try a different noise section</div>',
    },
    overSubCtrl: {
        t: 'Over-subtraction',
        b: '<h3>Over-subtraction Factor (α)</h3><p>Multiplier applied to the noise magnitude before subtraction:</p><div class="fm">|S_clean| = |S| - α × |N|</div><p><strong>α = 1.0:</strong> Subtracts exactly the estimated noise level. Starting point.</p><p><strong>α > 1.0:</strong> Subtracts more aggressively. Removes more noise but risks distorting the target signal.</p><p><strong>α < 1.0:</strong> Gentle subtraction. Some noise remains but less distortion risk.</p><div class="kp">Common range: 1.0 to 2.5. Start at 1.0, increase if noise persists, decrease if the target sounds distorted or "underwater."</div>',
    },
    specFloorCtrl: {
        t: 'Spectral Floor',
        b: '<h3>Spectral Floor (β)</h3><p>Minimum magnitude floor to prevent negative values after subtraction:</p><div class="fm">|S_clean| = max(|S| - α×|N|, β × |S|)</div><p>Without this floor, some frequency bins would go to zero or negative, creating artifacts called <strong>"musical noise"</strong> — random chirpy, tonal artifacts.</p><p><strong>β = 0.01:</strong> Low floor — more noise removed but risk of musical noise.</p><p><strong>β = 0.05-0.1:</strong> Higher floor — smoother result, some noise leaks through.</p><div class="kp">If you hear chirpy/watery artifacts, increase β. If too much noise remains, decrease β (and possibly increase α to compensate).</div>',
    },
    combOrderCtrl: {
        t: 'Processing Order',
        b: '<h3>Processing Order</h3><p>When combining spectral subtraction with band filtering, the order matters:</p><p><strong>Spectral Sub → Band Filter:</strong> First denoise the full signal, then extract frequency bands from the clean signal. Best for:</p><ul style="color:rgba(255,255,255,.5);font-size:11px;margin:4px 0 4px 16px"><li>Broadband noise (hiss, wind, HVAC hum)</li><li>When noise and target overlap in frequency</li></ul><p><strong>Band Filter → Spectral Sub:</strong> First isolate your bands of interest, then denoise within those bands. Best for:</p><ul style="color:rgba(255,255,255,.5);font-size:11px;margin:4px 0 4px 16px"><li>When target is in a specific frequency range</li><li>Reduces computational cost (denoising smaller signal)</li></ul>',
    },
    selRegion: {
        t: 'Waveform Selection',
        b: '<h3>Region Selection</h3><p>Drag on the waveform to select a time region. Playback (both original and extracted) will be confined to this region.</p><p><strong>Drag:</strong> Click and drag horizontally to select a region (shown with green edges and dimmed outside areas).</p><p><strong>Click:</strong> Single click to seek the playhead to that position.</p><p><strong>Double-click:</strong> Clear the selection (play full audio again).</p><div class="kp">This is powerful for isolating specific events — select just the moment a hammer strikes, then compare original vs extracted to hear how well your filter settings capture that specific sound.</div>',
    },
    multiBandCtrl: {
        t: 'Multi-Band Controls',
        b: '<h3>Multi-Band Filter Controls</h3><p>Each band defines an independent frequency range to extract:</p><p><strong>Low/High cutoff:</strong> Frequency boundaries of the band.</p><p><strong>Rolloff:</strong> Butterworth order for this specific band (each band can have different sharpness).</p><p><strong>+ Add Band:</strong> Create a new band with default settings.</p><p><strong>✕ Remove:</strong> Delete a band.</p><p>Bands are combined using <strong>maximum gain</strong> (union) at each frequency — overlapping bands don\'t amplify.</p><div class="kp">On the STFT spectrogram tab, you can also click-drag to create bands visually. Hold Shift to add bands instead of replacing.</div>',
    },
};
// ── EDUCATIONAL CONTENT (v6) ──
const EDU = {
    single: '<h3>Single-Band Extraction (Butterworth Rolloff)</h3><p>A bandpass filter passes frequencies within a range and attenuates those outside. Unlike the simple brick-wall approach (which abruptly zeros out bins), a Butterworth filter applies a smooth gain curve that gradually rolls off.</p><div class="fm">|H(f)| = 1 / sqrt(1 + (f/fc)^(2N))</div><p><strong>Rolloff order (N)</strong> controls the steepness of the transition:</p><div class="kp">Order 1: gentle slope (~20 dB/decade). Order 4: standard. Order 12: nearly brick-wall. Higher orders give sharper cutoffs but can introduce ringing artifacts on transient sounds.</div><p>The amber curve on the FFT plot shows the actual filter gain shape. Try adjusting the order while listening to extracted audio to hear the difference.</p>',
    multi: '<h3>Multi-Band Filtering</h3><p>Real-world sounds span multiple frequency regions. A hammer strike has a low-frequency thud (100-500 Hz) <em>and</em> a high-frequency crack (2-6 kHz). Single-band filtering can only capture one region at a time.</p><p>Multi-band mode lets you define several frequency bands, each with its own cutoff frequencies and rolloff. The combined output uses the <strong>maximum gain</strong> across all bands at each frequency bin (union, not sum), preventing amplification of overlapping regions.</p><div class="kp">Tip: On the STFT tab, click-drag to select a band visually. Hold Shift to add additional bands. Each band gets its own color on the spectrogram.</div>',
    spectral:
        '<h3>Spectral Subtraction</h3><p>Instead of selecting which frequencies to <em>keep</em>, spectral subtraction estimates the <em>noise</em> and removes it. The process:</p><p>1. Select a quiet section of audio (just background noise)<br>2. Capture its frequency profile<br>3. Subtract that profile from the entire signal</p><div class="fm">|S_clean(f)| = max( |S(f)| - α·|N(f)|, β·|S(f)| )</div><p><strong>Over-subtraction (α):</strong> Values above 1.0 remove more noise but risk distorting the target signal. Start at 1.0 and increase if noise remains.</p><p><strong>Spectral floor (β):</strong> Prevents the result from going below β × original magnitude. Too low → "musical noise" (chirpy artifacts from random residual bins). Too high → noise leaks through.</p><div class="kp">This is the same principle used in professional noise reduction (e.g., Audacity\'s noise reduction, iZotope RX). The quality depends heavily on having a good noise profile from a representative quiet section.</div>',
    combined:
        '<h3>Combined: Spectral Subtraction + Multi-Band</h3><p>Chain both techniques for maximum isolation power. The processing order matters:</p><p><strong>Spectral Sub → Band Filter:</strong> First clean noise from the entire signal, then extract your target frequency bands from the cleaned audio. Best when noise is broadband (hiss, hum, ambient rumble).</p><p><strong>Band Filter → Spectral Sub:</strong> First isolate your bands of interest, then apply noise reduction within those bands. Better when noise is concentrated in the same frequency region as your target sound.</p><div class="kp">For construction audio: try Spectral-first to remove ambient noise, then define bands around hammer-strike frequencies (200-800 Hz for thud, 2-5 kHz for impact crack).</div>',
};
let lCol = false;
function L(k) {
    const l = LS[k];
    if (!l) return;
    $('lT').textContent = l.t;
    $('lB').innerHTML = l.b;
    if (lCol) {
        lCol = false;
        $('lP').classList.remove('co');
    }
}
function togLP() {
    lCol = !lCol;
    $('lP').classList.toggle('co', lCol);
}
