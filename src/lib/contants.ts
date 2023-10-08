import { browser, dev } from '$app/environment';

export const ENDPOINT = dev
	? 'http://127.0.0.1:11434'
	: browser
	? 'http://127.0.0.1:11434'
	: 'http://host.docker.internal:11434';
