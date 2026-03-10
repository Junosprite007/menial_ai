<svelte:head>
	<title>Fourier Explorer | Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-6xl px-6 py-12">
	<div class="flex items-center justify-between mb-6">
		<div>
			<h1 class="text-3xl font-bold mb-2">Fourier Explorer</h1>
			<p class="text-text-dim">
				Browser-based FFT/STFT analysis with client-side Cooley-Tukey FFT, 4 isolation modes, and 50+ educational lessons.
			</p>
		</div>
		<a
			href="/explorer/"
			target="_blank"
			class="px-4 py-2 rounded-lg border border-border text-text-dim text-sm hover:border-accent/30 hover:text-text transition-colors"
		>
			Open Full Screen
		</a>
	</div>

	<div class="rounded-lg border border-border overflow-hidden">
		<iframe
			src="/explorer/"
			title="Fourier Explorer"
			class="w-full border-0"
			style="height: 80vh"
			sandbox="allow-scripts allow-same-origin allow-modals allow-popups allow-forms"
		></iframe>
	</div>

	<section class="mt-8 text-sm text-text-dim space-y-2">
		<h2 class="text-lg font-medium text-text mb-3">Features</h2>
		<ul class="grid md:grid-cols-2 gap-2">
			<li>Custom Cooley-Tukey FFT implementation</li>
			<li>Real-time STFT spectrogram visualization</li>
			<li>4 frequency isolation modes</li>
			<li>Multiple window functions (Hann, Hamming, Blackman)</li>
			<li>Interactive parameter controls</li>
			<li>50+ educational lessons with examples</li>
			<li>Audio recording and file upload</li>
			<li>Client-side processing (no server needed)</li>
		</ul>
	</section>
</div>
