<svelte:head>
	<title>Tools | Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-4xl px-6 py-12">
	<h1 class="text-3xl font-bold mb-2">Tools</h1>
	<p class="text-text-dim mb-8">Interactive demos of the audio analysis and signal processing tools built for this project.</p>

	<div class="grid md:grid-cols-2 gap-4">
		{#each [
			{ href: '/tools/classifier', title: 'Sound Classifier', desc: 'Upload audio and classify household sounds using a CNN trained on 50 ESC-50 classes with PANNs transfer learning.', tag: 'Interactive' },
			{ href: '/tools/fourier-fundamentals', title: 'Fourier Fundamentals', desc: 'Generate 5 educational figures exploring FFT, STFT, windowing, and the time-frequency tradeoff.', tag: 'Interactive' },
			{ href: '/tools/fourier-explorer', title: 'Fourier Explorer', desc: 'Browser-based FFT/STFT analysis with 4 isolation modes, client-side Cooley-Tukey FFT, and 50+ lessons.', tag: 'Client-side' },
			{ href: '/tools/isolator', title: 'Sound Isolator', desc: 'Decompose audio into components using Watershed segmentation, NMF source separation, or Connected Component Analysis.', tag: 'Interactive' },
		] as tool}
			<a href={tool.href} class="group block rounded-lg border border-border bg-surface p-6 transition-all hover:border-accent/30 hover:bg-surface-hover">
				<div class="flex items-start justify-between mb-3">
					<h2 class="text-lg font-medium text-text group-hover:text-accent transition-colors">{tool.title}</h2>
					<span class="text-xs rounded-full px-2 py-0.5 {tool.tag === 'Interactive' ? 'bg-green/15 text-green' : 'bg-accent-dim text-accent'}">{tool.tag}</span>
				</div>
				<p class="text-sm text-text-dim leading-relaxed">{tool.desc}</p>
			</a>
		{/each}
	</div>
</div>
