<svelte:head>
	<title>Sound Monitor | Menial AI</title>
</svelte:head>

<script lang="ts">
	import { onDestroy } from 'svelte';
	import {
		monitorStart,
		monitorChunk,
		monitorStop,
		type MonitorChunkResponse,
		type MonitorPrediction,
		type ActiveSound,
	} from '$lib/api';

	let running = $state(false);
	let sessionId = $state<string | null>(null);
	let sampleRate = $state(44100);
	let status = $state<'idle' | 'calibrating' | 'running' | 'error'>('idle');
	let calibrationProgress = $state(0);
	let errorMessage = $state('');
	let sensitivity = $state(0.4);
	let speakEnabled = $state(true);
	let lastSpoken: Record<string, number> = {};
	const SPEAK_COOLDOWN = 30_000;

	let predictions = $state<MonitorPrediction[]>([]);
	let activeSounds = $state<ActiveSound[]>([]);
	let messages = $state<string[]>([]);

	let audioContext: AudioContext | null = null;
	let mediaStream: MediaStream | null = null;
	let processorNode: ScriptProcessorNode | null = null;
	let sendInterval: ReturnType<typeof setInterval> | null = null;
	let audioBuffer: Float32Array[] = [];

	function speak(message: string) {
		if (!speakEnabled || !message) return;
		const now = Date.now();
		if (lastSpoken[message] && now - lastSpoken[message] < SPEAK_COOLDOWN) return;
		lastSpoken[message] = now;
		const utterance = new SpeechSynthesisUtterance(message);
		utterance.rate = 1.0;
		utterance.pitch = 1.0;
		speechSynthesis.speak(utterance);
	}

	let insecureContext = $state(
		typeof window !== 'undefined' && !window.isSecureContext
	);

	async function startMonitor() {
		errorMessage = '';
		try {
			if (!navigator.mediaDevices?.getUserMedia) {
				throw new Error(
					'Microphone access requires a secure context (HTTPS or localhost). ' +
					'Try accessing this page at http://localhost:3080/monitor instead.'
				);
			}

			mediaStream = await navigator.mediaDevices.getUserMedia({
				audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
			});

			const session = await monitorStart(sensitivity);
			sessionId = session.session_id;
			sampleRate = session.sample_rate;
			status = 'calibrating';
			running = true;

			audioContext = new AudioContext({ sampleRate });
			const source = audioContext.createMediaStreamSource(mediaStream);

			processorNode = audioContext.createScriptProcessor(4096, 1, 1);
			processorNode.onaudioprocess = (e) => {
				const data = e.inputBuffer.getChannelData(0);
				audioBuffer.push(new Float32Array(data));
			};
			source.connect(processorNode);
			processorNode.connect(audioContext.destination);

			sendInterval = setInterval(sendAudioChunk, 1000);
		} catch (err: any) {
			errorMessage = err.message || 'Failed to start monitor';
			status = 'error';
			running = false;
			cleanup();
		}
	}

	async function sendAudioChunk() {
		if (!sessionId || audioBuffer.length === 0) return;

		const totalLength = audioBuffer.reduce((sum, buf) => sum + buf.length, 0);
		const merged = new Float32Array(totalLength);
		let offset = 0;
		for (const buf of audioBuffer) {
			merged.set(buf, offset);
			offset += buf.length;
		}
		audioBuffer = [];

		try {
			const result: MonitorChunkResponse = await monitorChunk(sessionId, merged.buffer);

			if (result.status === 'calibrating') {
				status = 'calibrating';
				calibrationProgress = result.calibration_progress ?? 0;
			} else {
				status = 'running';
				predictions = result.predictions;
				activeSounds = result.active_sounds;
				if (result.messages.length > 0) {
					for (const msg of result.messages) speak(msg);
					messages = [...result.messages, ...messages].slice(0, 12);
				}
			}
		} catch (err: any) {
			if (err.message.includes('not found')) {
				await stopMonitor();
				errorMessage = 'Session expired. Please restart.';
				status = 'error';
			}
		}
	}

	async function stopMonitor() {
		running = false;
		status = 'idle';
		predictions = [];
		activeSounds = [];
		speechSynthesis.cancel();

		if (sessionId) {
			try {
				await monitorStop(sessionId);
			} catch {}
			sessionId = null;
		}
		cleanup();
	}

	function cleanup() {
		if (sendInterval) {
			clearInterval(sendInterval);
			sendInterval = null;
		}
		if (processorNode) {
			processorNode.disconnect();
			processorNode = null;
		}
		if (audioContext) {
			audioContext.close();
			audioContext = null;
		}
		if (mediaStream) {
			mediaStream.getTracks().forEach((t) => t.stop());
			mediaStream = null;
		}
		audioBuffer = [];
	}

	onDestroy(() => {
		if (running) stopMonitor();
	});

	function formatDuration(seconds: number): string {
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return m > 0 ? `${m}m ${s}s` : `${s}s`;
	}

	function confidenceColor(conf: number): string {
		if (conf > 0.6) return 'text-green';
		if (conf > 0.3) return 'text-amber';
		return 'text-text-muted';
	}

	function stateColor(state: string): string {
		if (state === 'active') return 'bg-green/20 text-green';
		if (state === 'detected') return 'bg-amber/20 text-amber';
		return 'bg-text-muted/20 text-text-muted';
	}
</script>

<div class="mx-auto max-w-4xl px-6 py-12">
	<h1 class="text-3xl font-bold mb-2">Household Sound Monitor</h1>
	<p class="text-text-dim mb-8">
		Real-time monitoring using your browser's microphone. Audio is captured locally, streamed to
		the backend for NMF cleaning, CNN classification, sound tracking, and contextual alerts.
	</p>

	<div class="flex items-center gap-4 mb-8">
		{#if !running}
			<button
				onclick={startMonitor}
				class="px-6 py-2.5 rounded-lg bg-brand text-white text-sm font-medium hover:bg-brand/90 transition-colors flex items-center gap-2"
			>
				<span class="inline-block w-3 h-3 rounded-full bg-red-500 animate-pulse"></span>
				Start Monitoring
			</button>

			<div class="flex items-center gap-2 text-sm text-text-dim">
				<span class="font-mono">Sensitivity:</span>
				<input type="range" min="0.1" max="1.0" step="0.05" bind:value={sensitivity} class="w-32 accent-brand" />
				<span class="font-mono w-10 text-accent">{sensitivity.toFixed(2)}</span>
			</div>

			<label class="flex items-center gap-2 text-sm text-text-dim cursor-pointer select-none">
				<input type="checkbox" bind:checked={speakEnabled} class="accent-brand" />
				Voice alerts
			</label>
		{:else}
			<button
				onclick={stopMonitor}
				class="px-6 py-2.5 rounded-lg border border-red-500/50 text-red-400 text-sm font-medium hover:bg-red-500/10 transition-colors"
			>
				Stop Monitoring
			</button>
			<label class="flex items-center gap-2 text-sm text-text-dim cursor-pointer select-none">
				<input type="checkbox" bind:checked={speakEnabled} class="accent-brand" />
				Voice
			</label>
			<span class="text-sm font-mono text-text-muted">
				{#if status === 'calibrating'}
					Calibrating noise profile...
				{:else}
					Listening
				{/if}
			</span>
		{/if}
	</div>

	{#if insecureContext}
		<div class="p-4 rounded-lg border border-amber/30 bg-amber/10 text-amber text-sm mb-8">
			Microphone access requires a secure context. Use
			<a href="http://localhost:3080/monitor" class="underline font-mono">http://localhost:3080/monitor</a>
			instead.
		</div>
	{/if}

	{#if errorMessage}
		<div class="p-4 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 text-sm mb-8">
			{errorMessage}
		</div>
	{/if}

	{#if status === 'calibrating'}
		<div class="mb-8">
			<p class="text-sm text-text-dim mb-2 font-mono">Calibrating noise profile — stay quiet for a moment...</p>
			<div class="w-full h-2 rounded-full bg-surface border border-border overflow-hidden">
				<div class="h-full bg-brand rounded-full transition-all duration-300" style="width: {calibrationProgress * 100}%"></div>
			</div>
		</div>
	{/if}

	{#if status === 'running'}
		<div class="grid gap-6 md:grid-cols-2 mb-8">
			<div class="rounded-lg border border-border bg-surface p-5">
				<h2 class="text-sm font-mono text-accent mb-4 uppercase tracking-wider">Current Detection</h2>
				{#if predictions.length > 0}
					<div class="space-y-2">
						{#each predictions as pred, i}
							<div class="flex items-center gap-3">
								<span class="font-mono text-sm w-40 truncate {i === 0 ? confidenceColor(pred.confidence) : 'text-text-dim'}">
									{pred.label}
								</span>
								<div class="flex-1 h-2 rounded-full bg-bg overflow-hidden">
									<div
										class="h-full rounded-full transition-all duration-300 {pred.confidence > 0.6 ? 'bg-green' : pred.confidence > 0.3 ? 'bg-amber' : 'bg-text-muted/50'}"
										style="width: {pred.confidence * 100}%"
									></div>
								</div>
								<span class="font-mono text-xs text-text-dim w-12 text-right">
									{(pred.confidence * 100).toFixed(1)}%
								</span>
							</div>
						{/each}
					</div>
				{:else}
					<p class="text-text-muted text-sm">Waiting for audio data...</p>
				{/if}
			</div>

			<div class="rounded-lg border border-border bg-surface p-5">
				<h2 class="text-sm font-mono text-accent mb-4 uppercase tracking-wider">Active Sounds</h2>
				{#if activeSounds.length > 0}
					<div class="space-y-2">
						{#each activeSounds as sound}
							<div class="flex items-center justify-between">
								<div class="flex items-center gap-2">
									<span class="text-xs rounded-full px-2 py-0.5 {stateColor(sound.state)}">{sound.state}</span>
									<span class="font-mono text-sm text-text">{sound.label}</span>
								</div>
								<span class="font-mono text-sm text-text-dim">{formatDuration(sound.duration)}</span>
							</div>
						{/each}
					</div>
				{:else}
					<p class="text-text-muted text-sm">No active sounds detected.</p>
				{/if}
			</div>
		</div>

		{#if messages.length > 0}
			<div class="rounded-lg border border-border bg-surface p-5 mb-8">
				<h2 class="text-sm font-mono text-accent mb-4 uppercase tracking-wider">Context Alerts</h2>
				<div class="space-y-2">
					{#each messages as msg}
						<div class="p-3 rounded border border-amber/20 bg-amber/5 text-sm font-serif italic text-text-dim">
							"{msg}"
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}

	<section class="mb-12 {status === 'running' ? 'opacity-50' : ''}">
		<h2 class="text-lg font-medium mb-4">Pipeline Architecture</h2>
		<div class="space-y-3 font-mono text-sm">
			{#each [
				{ label: 'Browser Mic Capture', desc: 'Web Audio API getUserMedia, float32 PCM at target sample rate' },
				{ label: 'Audio Streaming', desc: '~1-second chunks sent via HTTP POST to backend' },
				{ label: 'NMF Signal Cleaning', desc: 'STFT → NMF decomposition (4 components) → noise removal → iSTFT' },
				{ label: 'Feature Extraction', desc: '4-channel: Mel (128), MFCC (40→128), ZCR, STFT magnitude' },
				{ label: 'CNN Classification', desc: 'PANNsTransferModel or HouseholdSoundCNN → softmax → top-5' },
				{ label: 'Sound Tracking', desc: 'State machine: IDLE → DETECTED → ACTIVE → FADING → IDLE' },
				{ label: 'Context Engine', desc: 'Rule-based mapping: (sound, event, duration) → contextual response' },
			] as step, i}
				<div class="flex gap-4 items-start">
					<span class="text-accent w-6 text-right flex-shrink-0">{i + 1}.</span>
					<div class="flex-1">
						<span class="text-text">{step.label}</span>
						<span class="text-text-dim ml-2">{step.desc}</span>
					</div>
				</div>
			{/each}
		</div>
	</section>

	<section class="mb-12 {status === 'running' ? 'opacity-50' : ''}">
		<h2 class="text-lg font-medium mb-4">Context Engine Examples</h2>
		<div class="space-y-2 text-sm">
			{#each [
				{ sound: 'chopping', event: 'started', response: 'I hear chopping. Do you need the next recipe step?' },
				{ sound: 'boiling', event: '5 min', response: 'Your pot has been boiling for 5 minutes. Should I remind you to check it?' },
				{ sound: 'smoke_detector', event: 'started', response: 'ALERT: Smoke detector is going off! Please check immediately.' },
				{ sound: 'water_tap', event: '2 min', response: 'The water has been running for 2 minutes. Did you forget to turn it off?' },
				{ sound: 'crying_baby', event: '30 sec', response: 'The baby has been crying for 30 seconds.' },
			] as example}
				<div class="p-3 rounded-lg border border-border bg-surface">
					<div class="flex items-center gap-2 mb-1">
						<span class="text-xs font-mono text-accent">{example.sound}</span>
						<span class="text-text-muted text-xs">({example.event})</span>
					</div>
					<p class="text-text-dim font-serif italic">"{example.response}"</p>
				</div>
			{/each}
		</div>
	</section>
</div>
