import { dev, browser } from '$app/environment';
import { PUBLIC_API_BASE_URL } from '$env/static/public';

export const OLLAMA_API_BASE_URL =
	PUBLIC_API_BASE_URL === ''
		? dev
			? `http://${location.hostname}:8080/ollama/api`
			: browser
			? `http://${location.hostname}:11434/api`
			: `http://localhost:11434/api`
		: PUBLIC_API_BASE_URL;

export const WEBUI_API_BASE_URL = dev ? `http://${location.hostname}:8080/api/v1` : `/api/v1`;

export const WEB_UI_VERSION = 'v1.0.0-alpha-static';

// Source: https://kit.svelte.dev/docs/modules#$env-static-public
// This feature, akin to $env/static/private, exclusively incorporates environment variables
// that are prefixed with config.kit.env.publicPrefix (usually set to PUBLIC_).
// Consequently, these variables can be securely exposed to client-side code.

// Example of the .env configuration:
// OLLAMA_API_BASE_URL="http://localhost:11434/api"
// # Public
// PUBLIC_API_BASE_URL=$OLLAMA_API_BASE_URL
