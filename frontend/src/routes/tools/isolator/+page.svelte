<script lang="ts">
	import AudioUploader from '$lib/components/AudioUploader.svelte';
	import { runIsolator, type IsolatorResponse, type IsolatorParams } from '$lib/api';

	let result: IsolatorResponse | null = $state(null);
	let loading = $state(false);
	let error = $state('');
	let currentFile: File | null = $state(null);

	let params: IsolatorParams = $state({
		method: 'nmf',
		n_fft: 1024,
		overlap: 75,
		window: 'hann',
		max_freq: 8000,
		n_components: 5,
		threshold: 0.3,
		peak_dist: 50,
		min_size: 100,
		morph_size: 3,
		nmf_iter: 50,
	});

	async function handleFile(file: File) {
		currentFile = file;
		await analyze();
	}

	async function analyze() {
		if (!currentFile) return;
		loading = true;
		error = '';
		result = null;
		try {
			result = await runIsolator(currentFile, params);
		} catch (e: any) {
			error = e.message || 'Isolator failed';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Sound Isolator | Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-5xl px-6 py-12">
	<h1 class="text-3xl font-bold mb-2">Sound Isolator</h1>
	<p class="text-text-dim mb-8">
		Decompose audio into components using Watershed segmentation, NMF source separation,
		or Connected Component Analysis.
	</p>

	<AudioUploader onfile={handleFile} />

	{#if currentFile}
		<div class="mt-6 flex flex-wrap items-center gap-4">
			<label class="text-sm text-text-dim">
				Method
				<select bind:value={params.method} class="ml-2 bg-surface border border-border rounded px-2 py-1 text-text text-sm">
					<option value="nmf">NMF</option>
					<option value="watershed">Watershed</option>
					<option value="cca">CCA</option>
				</select>
			</label>

			<label class="text-sm text-text-dim">
				n_fft
				<select bind:value={params.n_fft} class="ml-2 bg-surface border border-border rounded px-2 py-1 text-text text-sm">
					{#each [256, 512, 1024, 2048, 4096] as v}
						<option value={v}>{v}</option>
					{/each}
				</select>
			</label>

			<label class="text-sm text-text-dim">
				Overlap
				<select bind:value={params.overlap} class="ml-2 bg-surface border border-border rounded px-2 py-1 text-text text-sm">
					<option value={50}>50%</option>
					<option value={75}>75%</option>
					<option value={87}>87%</option>
				</select>
			</label>

			{#if params.method === 'nmf'}
				<label class="text-sm text-text-dim">
					Components
					<input type="number" min="2" max="15" bind:value={params.n_components} class="ml-2 w-16 bg-surface border border-border rounded px-2 py-1 text-text text-sm" />
				</label>
			{/if}

			<button
				onclick={analyze}
				class="px-4 py-1.5 rounded-lg bg-brand text-white text-sm font-medium hover:bg-brand/90 transition-colors"
			>
				Re-analyze
			</button>
		</div>
	{/if}

	{#if loading}
		<div class="mt-8 text-center">
			<div class="inline-block w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin"></div>
			<p class="text-text-dim text-sm mt-2">Decomposing audio...</p>
		</div>
	{/if}

	{#if error}
		<div class="mt-8 p-4 rounded-lg border border-rose/30 bg-rose/10 text-rose text-sm">{error}</div>
	{/if}

	{#if result}
		<div class="mt-8">
			<div class="mb-6">
				<h2 class="text-lg font-medium mb-2">Full Spectrogram</h2>
				<img src={result.spectrogram_url} alt="Full spectrogram" class="w-full rounded-lg border border-border" />
			</div>

			<h2 class="text-lg font-medium mb-4">Components ({result.num_components})</h2>
			<div class="grid md:grid-cols-2 gap-4">
				{#each result.components as comp}
					<div class="rounded-lg border border-border bg-surface p-4">
						<div class="flex items-center justify-between mb-2">
							<span class="text-sm font-medium" style="color: {comp.color}">Component {comp.id}</span>
							<span class="text-xs text-text-dim font-mono">{comp.freq_peak.toFixed(0)} Hz</span>
						</div>
						<img src={comp.spectrogram_url} alt="Component {comp.id}" class="w-full rounded mb-2" />
						{#if comp.audio_url}
							<audio controls src={comp.audio_url} class="w-full h-8"></audio>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<section class="mt-16 border-t border-border pt-8">
		<h2 class="text-lg font-medium mb-4">Decomposition Methods</h2>
		<div class="grid md:grid-cols-3 gap-6 text-sm text-text-dim">
			<div>
				<h3 class="text-text font-medium mb-2">NMF</h3>
				<p>Non-Negative Matrix Factorization decomposes the spectrogram into a set of basis vectors and activations, naturally separating additive sources.</p>
			</div>
			<div>
				<h3 class="text-text font-medium mb-2">Watershed</h3>
				<p>Treats the spectrogram as a topographic surface and segments peaks into basins, identifying distinct time-frequency regions.</p>
			</div>
			<div>
				<h3 class="text-text font-medium mb-2">CCA</h3>
				<p>Connected Component Analysis thresholds the spectrogram and groups connected regions of energy into separate components.</p>
			</div>
		</div>
	</section>
</div>
