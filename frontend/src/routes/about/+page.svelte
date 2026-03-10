<svelte:head>
	<title>About | Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-4xl px-6 py-12">
	<h1 class="text-3xl font-bold mb-2">About Menial AI</h1>
	<p class="text-text-dim mb-12">
		A real-time household sound classification and audio signal processing toolkit,
		built for TECHIN 513A at the University of Washington.
	</p>

	<section class="mb-12">
		<h2 class="text-lg font-medium mb-2">Authors</h2>
		<p class="text-text-dim">Joshua Kirby &amp; Alan Nur</p>
		<p class="text-text-muted text-sm">TECHIN 513A — Managing Data And Signal Processing, University of Washington</p>
	</section>

	<section class="mb-12">
		<h2 class="text-lg font-medium mb-4">Model Performance</h2>
		<div class="overflow-x-auto">
			<table class="w-full text-sm">
				<thead>
					<tr class="border-b border-border text-left text-text-dim">
						<th class="py-2 pr-4">Metric</th>
						<th class="py-2 pr-4">Custom CNN</th>
						<th class="py-2">PANNs Transfer</th>
					</tr>
				</thead>
				<tbody class="text-text-dim">
					<tr class="border-b border-border/50">
						<td class="py-2 pr-4 text-text">Test Accuracy</td>
						<td class="py-2 pr-4 font-mono">~59%</td>
						<td class="py-2 font-mono text-accent">74-85%</td>
					</tr>
					<tr class="border-b border-border/50">
						<td class="py-2 pr-4 text-text">Parameters</td>
						<td class="py-2 pr-4 font-mono">110K</td>
						<td class="py-2 font-mono">7.7M</td>
					</tr>
					<tr class="border-b border-border/50">
						<td class="py-2 pr-4 text-text">Model Size</td>
						<td class="py-2 pr-4 font-mono">1.2 MB</td>
						<td class="py-2 font-mono">312 MB</td>
					</tr>
					<tr class="border-b border-border/50">
						<td class="py-2 pr-4 text-text">Inference Speed</td>
						<td class="py-2 pr-4 font-mono">~20ms</td>
						<td class="py-2 font-mono">~45ms</td>
					</tr>
					<tr>
						<td class="py-2 pr-4 text-text">Training Time</td>
						<td class="py-2 pr-4 font-mono">90-120 min</td>
						<td class="py-2 font-mono">60-90 min</td>
					</tr>
				</tbody>
			</table>
		</div>
	</section>

	<section class="mb-12">
		<h2 class="text-lg font-medium mb-4">DSP Concepts Implemented</h2>
		<div class="flex flex-wrap gap-2">
			{#each [
				'FFT (Cooley-Tukey)', 'STFT', 'Mel Spectrograms', 'MFCC',
				'NMF Source Separation', 'Watershed Segmentation', 'Connected Component Analysis',
				'Butterworth Filtering', 'Spectral Subtraction', 'Window Functions',
				'SpecAugment', 'Zero-Crossing Rate', 'Amplitude-to-dB', 'Adaptive Pooling',
			] as concept}
				<span class="px-3 py-1 rounded-full border border-border bg-surface text-xs text-text-dim font-mono">{concept}</span>
			{/each}
		</div>
	</section>

	<section class="mb-12">
		<h2 class="text-lg font-medium mb-4">Technology Stack</h2>
		<div class="grid md:grid-cols-2 gap-4 text-sm">
			{#each [
				{ cat: 'ML & Training', items: 'PyTorch, torchaudio, PANNs CNN14, SpecAugment' },
				{ cat: 'DSP', items: 'NumPy, SciPy, librosa, custom FFT/STFT' },
				{ cat: 'Web Backend', items: 'FastAPI, uvicorn, Python 3.12' },
				{ cat: 'Web Frontend', items: 'SvelteKit, Tailwind CSS v4, TypeScript' },
				{ cat: 'Audio I/O', items: 'sounddevice, Web Audio API, ffmpeg' },
				{ cat: 'Deployment', items: 'Docker, nginx-proxy, adapter-node' },
			] as stack}
				<div class="rounded-lg border border-border bg-surface p-4">
					<h3 class="text-text font-medium mb-1">{stack.cat}</h3>
					<p class="text-text-dim">{stack.items}</p>
				</div>
			{/each}
		</div>
	</section>

	<section>
		<h2 class="text-lg font-medium mb-4">Codebase</h2>
		<div class="flex gap-8 text-sm text-text-dim">
			<div>
				<span class="text-2xl font-bold text-accent font-mono">~3,300</span>
				<span class="ml-2">lines of Python</span>
			</div>
			<div>
				<span class="text-2xl font-bold text-accent font-mono">~2,900</span>
				<span class="ml-2">lines of JavaScript</span>
			</div>
		</div>
	</section>
</div>
