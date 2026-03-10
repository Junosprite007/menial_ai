<script lang="ts">
	import AudioUploader from '$lib/components/AudioUploader.svelte';
	import { classify, type ClassifyResponse } from '$lib/api';

	let result: ClassifyResponse | null = $state(null);
	let loading = $state(false);
	let error = $state('');
	let fileName = $state('');

	async function handleFile(file: File) {
		loading = true;
		error = '';
		result = null;
		fileName = file.name;

		try {
			result = await classify(file);
		} catch (e: any) {
			error = e.message || 'Classification failed';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Sound Classifier | Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-3xl px-6 py-12">
	<h1 class="text-3xl font-bold mb-2">Sound Classifier</h1>
	<p class="text-text-dim mb-8">
		Upload an audio file to classify household sounds using our trained CNN model.
		In production, this runs in real-time via microphone. Here, you upload a file for classification.
	</p>

	<AudioUploader onfile={handleFile} />

	{#if loading}
		<div class="mt-8 text-center">
			<div class="inline-block w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin"></div>
			<p class="text-text-dim text-sm mt-2">Classifying {fileName}...</p>
		</div>
	{/if}

	{#if error}
		<div class="mt-8 p-4 rounded-lg border border-rose/30 bg-rose/10 text-rose text-sm">{error}</div>
	{/if}

	{#if result}
		<div class="mt-8">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-lg font-medium">Results</h2>
				<span class="text-xs text-text-dim font-mono">model: {result.model_type} ({result.num_classes} classes)</span>
			</div>

			<div class="space-y-3">
				{#each result.predictions as pred, i}
					{@const pct = (pred.confidence * 100)}
					{@const isTop = i === 0}
					<div class="flex items-center gap-4">
						<span class="w-40 text-sm truncate {isTop ? 'text-accent font-medium' : 'text-text-dim'}">{pred.label}</span>
						<div class="flex-1 h-6 rounded bg-surface overflow-hidden">
							<div
								class="h-full rounded transition-all duration-500 {isTop ? 'bg-accent' : 'bg-accent/30'}"
								style="width: {pct}%"
							></div>
						</div>
						<span class="w-16 text-right text-sm font-mono {isTop ? 'text-accent' : 'text-text-dim'}">{pct.toFixed(1)}%</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<section class="mt-16 border-t border-border pt-8">
		<h2 class="text-lg font-medium mb-4">How It Works</h2>
		<div class="grid md:grid-cols-2 gap-6 text-sm text-text-dim">
			<div>
				<h3 class="text-text font-medium mb-2">Custom CNN (Baseline)</h3>
				<ul class="space-y-1">
					<li>4-block convolutional architecture</li>
					<li>110K parameters, 1.2 MB model</li>
					<li>4-channel input: Mel, MFCC, ZCR, STFT</li>
					<li>~59% accuracy on ESC-50</li>
				</ul>
			</div>
			<div>
				<h3 class="text-text font-medium mb-2">PANNs Transfer Learning</h3>
				<ul class="space-y-1">
					<li>Pre-trained CNN14 backbone (AudioSet 2M+ clips)</li>
					<li>7.7M parameters, 312 MB model</li>
					<li>Two-phase training with differential learning rates</li>
					<li>74-85% accuracy on ESC-50</li>
				</ul>
			</div>
		</div>
	</section>
</div>
