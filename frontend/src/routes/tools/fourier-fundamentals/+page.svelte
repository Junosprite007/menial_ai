<script lang="ts">
	import AudioUploader from '$lib/components/AudioUploader.svelte';
	import { generateFourierFigures, type FourierResponse } from '$lib/api';

	let result: FourierResponse | null = $state(null);
	let loading = $state(false);
	let error = $state('');

	const figureLabels: Record<string, string> = {
		'time_domain': 'Time Domain — Raw waveform of the audio signal',
		'fft_spectrum': 'FFT Spectrum — Frequency content via Fast Fourier Transform',
		'fft_windowing': 'FFT Windowing & Zero-padding — Effect of window functions',
		'stft_spectrogram': 'STFT Spectrogram — Time-frequency representation',
		'time_freq_tradeoff': 'Time-Frequency Tradeoff — Comparing different n_fft values',
	};

	async function handleFile(file: File) {
		loading = true;
		error = '';
		result = null;
		try {
			result = await generateFourierFigures(file);
		} catch (e: any) {
			error = e.message || 'Figure generation failed';
		} finally {
			loading = false;
		}
	}

	async function useSynthetic() {
		loading = true;
		error = '';
		result = null;
		try {
			result = await generateFourierFigures();
		} catch (e: any) {
			error = e.message || 'Figure generation failed';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Fourier Fundamentals | Menial AI</title>
</svelte:head>

<div class="mx-auto max-w-4xl px-6 py-12">
	<h1 class="text-3xl font-bold mb-2">Fourier Fundamentals</h1>
	<p class="text-text-dim mb-8">
		Upload audio or use a synthetic signal to generate 5 educational figures exploring
		FFT, STFT, windowing, and the time-frequency tradeoff.
	</p>

	<div class="flex gap-4 items-start mb-8">
		<div class="flex-1">
			<AudioUploader onfile={handleFile} />
		</div>
		<div class="text-text-dim text-sm pt-8">or</div>
		<button
			onclick={useSynthetic}
			class="mt-6 px-6 py-2.5 rounded-lg border border-border text-text-dim text-sm hover:border-accent/30 hover:text-text transition-colors"
		>
			Use Synthetic Demo
		</button>
	</div>

	{#if loading}
		<div class="mt-8 text-center">
			<div class="inline-block w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin"></div>
			<p class="text-text-dim text-sm mt-2">Generating figures...</p>
		</div>
	{/if}

	{#if error}
		<div class="mt-8 p-4 rounded-lg border border-rose/30 bg-rose/10 text-rose text-sm">{error}</div>
	{/if}

	{#if result}
		<div class="mt-8 space-y-8">
			<p class="text-sm text-text-dim">Source: {result.source}</p>
			{#each result.figures as fig}
				<div class="rounded-lg border border-border bg-surface p-4">
					<h3 class="text-sm font-medium mb-3 text-accent">
						{figureLabels[fig.name] || fig.name}
					</h3>
					<img src={fig.url} alt={fig.name} class="w-full rounded" />
				</div>
			{/each}
		</div>
	{/if}
</div>
