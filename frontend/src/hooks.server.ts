import type { Handle } from '@sveltejs/kit';

const API_URL = process.env.API_URL || 'http://localhost:8000';

export const handle: Handle = async ({ event, resolve }) => {
	const path = event.url.pathname;

	// Proxy /api/* and /explorer/* requests to the backend
	if (path.startsWith('/api/') || path.startsWith('/explorer/') || path.startsWith('/explorer') || path === '/analyze') {
		const target = `${API_URL}${path}${event.url.search}`;
		const headers = new Headers(event.request.headers);
		headers.delete('host');

		const response = await fetch(target, {
			method: event.request.method,
			headers,
			body: event.request.method !== 'GET' && event.request.method !== 'HEAD'
				? await event.request.arrayBuffer()
				: undefined,
			// @ts-ignore - duplex needed for streaming
			duplex: 'half',
		});

		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: response.headers,
		});
	}

	return resolve(event);
};
