import { browser } from '$app/environment';
import { PUBLIC_API_ENDPOINT } from '$env/static/public';

export const API_ENDPOINT =
	PUBLIC_API_ENDPOINT === ''
		? browser
			? `http://${location.hostname}:11434/api`
			: `http://localhost:11434/api`
		: PUBLIC_API_ENDPOINT;

// Source: https://kit.svelte.dev/docs/modules#$env-static-public
// This feature, akin to $env/static/private, exclusively incorporates environment variables
// that are prefixed with config.kit.env.publicPrefix (usually set to PUBLIC_).
// Consequently, these variables can be securely exposed to client-side code.

// Example of the .env configuration:
// OLLAMA_API_ENDPOINT="http://localhost:11434/api"
// # Public
// PUBLIC_API_ENDPOINT=$OLLAMA_API_ENDPOINT
