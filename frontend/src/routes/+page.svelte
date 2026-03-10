<svelte:head>
	<title>Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-6xl px-6 py-20">
	<section class="text-center mb-20">
		<h1 class="text-5xl font-bold mb-4">
			<span class="text-accent">Menial</span><span class="text-text-dim">_</span><span class="text-text">AI</span>
		</h1>
		<p class="text-xl text-text-dim max-w-2xl mx-auto mb-8">
			Real-time household sound classification and audio signal processing toolkit
		</p>
		<div class="flex justify-center gap-4">
			<a href="/tools" class="px-6 py-2.5 rounded-lg bg-brand text-white text-sm font-medium hover:bg-brand/90 transition-colors">Explore Tools</a>
			<a href="/about" class="px-6 py-2.5 rounded-lg border border-border text-text-dim text-sm hover:border-accent/30 hover:text-text transition-colors">Learn More</a>
		</div>
	</section>

	<section class="mb-20">
		<h2 class="text-sm font-mono text-accent mb-6 text-center uppercase tracking-wider">Audio Pipeline</h2>
		<div class="flex flex-wrap items-center justify-center gap-3 font-mono text-sm">
			{#each ['Microphone', 'NMF Clean', 'Mel Spectrogram', 'CNN Classify', 'Sound Track', 'Context Engine', 'Voice Alert'] as step, i}
				{#if i > 0}<span class="text-text-muted">&rarr;</span>{/if}
				<span class="px-3 py-1.5 rounded border border-border bg-surface text-text-dim">{step}</span>
			{/each}
		</div>
	</section>

	<section class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-20">
		{#each [
			{ value: '74-85%', label: 'Classification Accuracy' },
			{ value: '50', label: 'Sound Classes' },
			{ value: '~45ms', label: 'Inference Time' },
			{ value: '100%', label: 'Local Processing' },
		] as stat}
			<div class="text-center p-4 rounded-lg border border-border bg-surface">
				<div class="text-2xl font-bold text-accent font-mono">{stat.value}</div>
				<div class="text-xs text-text-dim mt-1">{stat.label}</div>
			</div>
		{/each}
	</section>

	<section>
		<h2 class="text-sm font-mono text-accent mb-6 text-center uppercase tracking-wider">Tools</h2>
		<div class="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each [
				{ href: '/tools/classifier', title: 'Sound Classifier', desc: 'Upload audio and classify household sounds using a CNN trained on 50 ESC-50 classes with PANNs transfer learning.', tag: 'Interactive', tagColor: 'bg-green/15 text-green' },
				{ href: '/tools/fourier-fundamentals', title: 'Fourier Fundamentals', desc: 'Generate 5 educational figures exploring FFT, STFT, windowing, and the time-frequency tradeoff.', tag: 'Interactive', tagColor: 'bg-green/15 text-green' },
				{ href: '/tools/fourier-explorer', title: 'Fourier Explorer', desc: 'Browser-based FFT/STFT analysis with 4 isolation modes, client-side Cooley-Tukey FFT, and 50+ lessons.', tag: 'Client-side', tagColor: 'bg-accent-dim text-accent' },
				{ href: '/tools/isolator', title: 'Sound Isolator', desc: 'Decompose audio into components using Watershed segmentation, NMF source separation, or Connected Component Analysis.', tag: 'Interactive', tagColor: 'bg-green/15 text-green' },
				{ href: '/monitor', title: 'Sound Monitor', desc: 'Full real-time monitoring pipeline with NMF cleaning, sound tracking, and context-aware voice alerts.', tag: 'Interactive', tagColor: 'bg-green/15 text-green' },
				{ href: '/about', title: 'About', desc: 'Project architecture, DSP concepts, ML results, and technical documentation.', tag: 'Documentation', tagColor: 'bg-accent-dim text-accent' },
			] as tool}
				<a href={tool.href} class="group block rounded-lg border border-border bg-surface p-6 transition-all hover:border-accent/30 hover:bg-surface-hover">
					<div class="flex items-start justify-between mb-3">
						<h3 class="text-lg font-medium text-text group-hover:text-accent transition-colors">{tool.title}</h3>
						<span class="text-xs rounded-full px-2 py-0.5 {tool.tagColor}">{tool.tag}</span>
					</div>
					<p class="text-sm text-text-dim leading-relaxed">{tool.desc}</p>
				</a>
			{/each}
		</div>
	</section>
</div>
