import { browser, dev } from '$app/environment';

export const ENDPOINT =
	process.env.OLLAMA_ENDPOINT != undefined
		? process.env.OLLAMA_ENDPOINT
		: browser
		? `http://${location.hostname}:11434`
		: dev
		? 'http://127.0.0.1:11434'
		: 'http://host.docker.internal:11434';
