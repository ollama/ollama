import { dev } from '$app/environment';

export const WEBUI_BASE_URL = dev ? `http://${location.hostname}:8080` : ``;

export const WEBUI_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1`;
export const OLLAMA_API_BASE_URL = `${WEBUI_BASE_URL}/ollama/api`;
export const OPENAI_API_BASE_URL = `${WEBUI_BASE_URL}/openai/api`;
export const RAG_API_BASE_URL = `${WEBUI_BASE_URL}/rag/api/v1`;

export const WEB_UI_VERSION = 'v1.0.0-alpha-static';

export const REQUIRED_OLLAMA_VERSION = '0.1.16';

export const SUPPORTED_FILE_TYPE = [
	'application/epub+zip',
	'application/pdf',
	'text/plain',
	'text/csv',
	'text/xml',
	'text/x-python',
	'text/css',
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
	'application/octet-stream',
	'application/x-javascript',
	'text/markdown'
];

export const SUPPORTED_FILE_EXTENSIONS = [
	'md', 'rst','go', 'py', 'java', 'sh', 'bat', 'ps1', 'cmd', 'js', 
	'ts', 'css', 'cpp', 'hpp','h', 'c', 'cs', 'sql', 'log', 'ini',
	'pl', 'pm', 'r', 'dart', 'dockerfile', 'env', 'php', 'hs',
	'hsc', 'lua', 'nginxconf', 'conf', 'm', 'mm', 'plsql', 'perl',
	'rb', 'rs', 'db2', 'scala', 'bash', 'swift', 'vue', 'svelte',
	'doc','docx', 'pdf', 'csv', 'txt', 'xls', 'xlsx'
];

// Source: https://kit.svelte.dev/docs/modules#$env-static-public
// This feature, akin to $env/static/private, exclusively incorporates environment variables
// that are prefixed with config.kit.env.publicPrefix (usually set to PUBLIC_).
// Consequently, these variables can be securely exposed to client-side code.

// Example of the .env configuration:
// OLLAMA_API_BASE_URL="http://localhost:11434/api"
// # Public
// PUBLIC_API_BASE_URL=$OLLAMA_API_BASE_URL
