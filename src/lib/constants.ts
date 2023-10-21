import { browser, dev } from '$app/environment';

export const API_ENDPOINT = browser
	? `https://localhost/api`
	: dev
	? `http://localhost:11434/api`
	: 'http://host.docker.internal:11434/api';
