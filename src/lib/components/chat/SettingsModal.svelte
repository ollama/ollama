<script lang="ts">
	import Modal from '../common/Modal.svelte';

	import {
		WEB_UI_VERSION,
		OLLAMA_API_BASE_URL,
		WEBUI_API_BASE_URL,
		WEBUI_BASE_URL
	} from '$lib/constants';
	import toast from 'svelte-french-toast';
	import { onMount } from 'svelte';
	import { config, info, models, settings, user } from '$lib/stores';
	import { splitStream, getGravatarURL } from '$lib/utils';
	import Advanced from './Settings/Advanced.svelte';
	import { stringify } from 'postcss';

	export let show = false;

	const saveSettings = async (updated) => {
		console.log(updated);
		await settings.set({ ...$settings, ...updated });
		await models.set(await getModels());
		localStorage.setItem('settings', JSON.stringify($settings));
	};

	let selectedTab = 'general';

	// General
	let API_BASE_URL = OLLAMA_API_BASE_URL;
	let theme = 'dark';
	let notificationEnabled = false;
	let system = '';

	// Advanced
	let requestFormat = '';
	let options = {
		// Advanced
		seed: 0,
		temperature: '',
		repeat_penalty: '',
		repeat_last_n: '',
		mirostat: '',
		mirostat_eta: '',
		mirostat_tau: '',
		top_k: '',
		top_p: '',
		stop: '',
		tfs_z: '',
		num_ctx: ''
	};

	// Models
	let modelTransferring = false;

	let modelTag = '';
	let digest = '';
	let pullProgress = null;

	let modelUploadMode = 'file';
	let modelInputFile = '';
	let modelFileUrl = '';
	let modelFileContent = `TEMPLATE """{{ .System }}\nUSER: {{ .Prompt }}\nASSSISTANT: """\nPARAMETER num_ctx 4096\nPARAMETER stop "</s>"\nPARAMETER stop "USER:"\nPARAMETER stop "ASSSISTANT:"`;
	let modelFileDigest = '';
	let uploadProgress = null;

	let deleteModelTag = '';

	// Addons
	let titleAutoGenerate = true;
	let speechAutoSend = false;
	let responseAutoCopy = false;

	let gravatarEmail = '';
	let OPENAI_API_KEY = '';
	let OPENAI_API_BASE_URL = '';

	// Auth
	let authEnabled = false;
	let authType = 'Basic';
	let authContent = '';

	const checkOllamaConnection = async () => {
		if (API_BASE_URL === '') {
			API_BASE_URL = OLLAMA_API_BASE_URL;
		}
		const _models = await getModels(API_BASE_URL, 'ollama');

		if (_models.length > 0) {
			toast.success('Server connection verified');
			await models.set(_models);

			saveSettings({
				API_BASE_URL: API_BASE_URL
			});
		}
	};

	const toggleTheme = async () => {
		if (theme === 'dark') {
			theme = 'light';
		} else {
			theme = 'dark';
		}

		localStorage.theme = theme;

		document.documentElement.classList.remove(theme === 'dark' ? 'light' : 'dark');
		document.documentElement.classList.add(theme);
	};

	const toggleRequestFormat = async () => {
		if (requestFormat === '') {
			requestFormat = 'json';
		} else {
			requestFormat = '';
		}

		saveSettings({ requestFormat: requestFormat !== '' ? requestFormat : undefined });
	};

	const toggleSpeechAutoSend = async () => {
		speechAutoSend = !speechAutoSend;
		saveSettings({ speechAutoSend: speechAutoSend });
	};

	const toggleTitleAutoGenerate = async () => {
		titleAutoGenerate = !titleAutoGenerate;
		saveSettings({ titleAutoGenerate: titleAutoGenerate });
	};

	const toggleNotification = async () => {
		const permission = await Notification.requestPermission();

		if (permission === 'granted') {
			notificationEnabled = !notificationEnabled;
			saveSettings({ notificationEnabled: notificationEnabled });
		} else {
			toast.error(
				'Response notifications cannot be activated as the website permissions have been denied. Please visit your browser settings to grant the necessary access.'
			);
		}
	};

	const toggleResponseAutoCopy = async () => {
		const permission = await navigator.clipboard
			.readText()
			.then(() => {
				return 'granted';
			})
			.catch(() => {
				return '';
			});

		console.log(permission);

		if (permission === 'granted') {
			responseAutoCopy = !responseAutoCopy;
			saveSettings({ responseAutoCopy: responseAutoCopy });
		} else {
			toast.error(
				'Clipboard write permission denied. Please check your browser settings to grant the necessary access.'
			);
		}
	};

	const toggleAuthHeader = async () => {
		authEnabled = !authEnabled;
	};

	const pullModelHandler = async () => {
		modelTransferring = true;
		const res = await fetch(`${API_BASE_URL}/pull`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream',
				...($settings.authHeader && { Authorization: $settings.authHeader }),
				...($user && { Authorization: `Bearer ${localStorage.token}` })
			},
			body: JSON.stringify({
				name: modelTag
			})
		});

		const reader = res.body
			.pipeThrough(new TextDecoderStream())
			.pipeThrough(splitStream('\n'))
			.getReader();

		while (true) {
			const { value, done } = await reader.read();
			if (done) break;

			try {
				let lines = value.split('\n');

				for (const line of lines) {
					if (line !== '') {
						console.log(line);
						let data = JSON.parse(line);
						console.log(data);

						if (data.error) {
							throw data.error;
						}

						if (data.detail) {
							throw data.detail;
						}
						if (data.status) {
							if (!data.digest) {
								toast.success(data.status);

								if (data.status === 'success') {
									const notification = new Notification(`Ollama`, {
										body: `Model '${modelTag}' has been successfully downloaded.`,
										icon: '/favicon.png'
									});
								}
							} else {
								digest = data.digest;
								if (data.completed) {
									pullProgress = Math.round((data.completed / data.total) * 1000) / 10;
								} else {
									pullProgress = 100;
								}
							}
						}
					}
				}
			} catch (error) {
				console.log(error);
				toast.error(error);
			}
		}

		modelTag = '';
		modelTransferring = false;

		models.set(await getModels());
	};

	const calculateSHA256 = async (file) => {
		console.log(file);
		// Create a FileReader to read the file asynchronously
		const reader = new FileReader();

		// Define a promise to handle the file reading
		const readFile = new Promise((resolve, reject) => {
			reader.onload = () => resolve(reader.result);
			reader.onerror = reject;
		});

		// Read the file as an ArrayBuffer
		reader.readAsArrayBuffer(file);

		try {
			// Wait for the FileReader to finish reading the file
			const buffer = await readFile;

			// Convert the ArrayBuffer to a Uint8Array
			const uint8Array = new Uint8Array(buffer);

			// Calculate the SHA-256 hash using Web Crypto API
			const hashBuffer = await crypto.subtle.digest('SHA-256', uint8Array);

			// Convert the hash to a hexadecimal string
			const hashArray = Array.from(new Uint8Array(hashBuffer));
			const hashHex = hashArray.map((byte) => byte.toString(16).padStart(2, '0')).join('');

			return `sha256:${hashHex}`;
		} catch (error) {
			console.error('Error calculating SHA-256 hash:', error);
			throw error;
		}
	};

	const uploadModelHandler = async () => {
		modelTransferring = true;
		uploadProgress = 0;

		let uploaded = false;
		let fileResponse = null;
		let name = '';

		if (modelUploadMode === 'file') {
			const file = modelInputFile[0];
			const formData = new FormData();
			formData.append('file', file);

			fileResponse = await fetch(`${WEBUI_API_BASE_URL}/utils/upload`, {
				method: 'POST',
				headers: {
					...($settings.authHeader && { Authorization: $settings.authHeader }),
					...($user && { Authorization: `Bearer ${localStorage.token}` })
				},
				body: formData
			}).catch((error) => {
				console.log(error);
				return null;
			});
		} else {
			fileResponse = await fetch(`${WEBUI_API_BASE_URL}/utils/download?url=${modelFileUrl}`, {
				method: 'GET',
				headers: {
					...($settings.authHeader && { Authorization: $settings.authHeader }),
					...($user && { Authorization: `Bearer ${localStorage.token}` })
				}
			}).catch((error) => {
				console.log(error);
				return null;
			});
		}

		if (fileResponse && fileResponse.ok) {
			const reader = fileResponse.body
				.pipeThrough(new TextDecoderStream())
				.pipeThrough(splitStream('\n'))
				.getReader();

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;

				try {
					let lines = value.split('\n');

					for (const line of lines) {
						if (line !== '') {
							let data = JSON.parse(line.replace(/^data: /, ''));

							if (data.progress) {
								uploadProgress = data.progress;
							}

							if (data.error) {
								throw data.error;
							}

							if (data.done) {
								modelFileDigest = data.blob;
								name = data.name;
								uploaded = true;
							}
						}
					}
				} catch (error) {
					console.log(error);
				}
			}
		}

		if (uploaded) {
			const res = await fetch(`${$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/create`, {
				method: 'POST',
				headers: {
					'Content-Type': 'text/event-stream',
					...($settings.authHeader && { Authorization: $settings.authHeader }),
					...($user && { Authorization: `Bearer ${localStorage.token}` })
				},
				body: JSON.stringify({
					name: `${name}:latest`,
					modelfile: `FROM @${modelFileDigest}\n${modelFileContent}`
				})
			}).catch((err) => {
				console.log(err);
				return null;
			});

			if (res && res.ok) {
				const reader = res.body
					.pipeThrough(new TextDecoderStream())
					.pipeThrough(splitStream('\n'))
					.getReader();

				while (true) {
					const { value, done } = await reader.read();
					if (done) break;

					try {
						let lines = value.split('\n');

						for (const line of lines) {
							if (line !== '') {
								console.log(line);
								let data = JSON.parse(line);
								console.log(data);

								if (data.error) {
									throw data.error;
								}
								if (data.detail) {
									throw data.detail;
								}

								if (data.status) {
									if (
										!data.digest &&
										!data.status.includes('writing') &&
										!data.status.includes('sha256')
									) {
										toast.success(data.status);
									} else {
										if (data.digest) {
											digest = data.digest;

											if (data.completed) {
												pullProgress = Math.round((data.completed / data.total) * 1000) / 10;
											} else {
												pullProgress = 100;
											}
										}
									}
								}
							}
						}
					} catch (error) {
						console.log(error);
						toast.error(error);
					}
				}
			}
		}

		modelFileUrl = '';
		modelInputFile = '';
		modelTransferring = false;
		uploadProgress = null;

		models.set(await getModels());
	};

	const deleteModelHandler = async () => {
		const res = await fetch(`${API_BASE_URL}/delete`, {
			method: 'DELETE',
			headers: {
				'Content-Type': 'text/event-stream',
				...($settings.authHeader && { Authorization: $settings.authHeader }),
				...($user && { Authorization: `Bearer ${localStorage.token}` })
			},
			body: JSON.stringify({
				name: deleteModelTag
			})
		});

		const reader = res.body
			.pipeThrough(new TextDecoderStream())
			.pipeThrough(splitStream('\n'))
			.getReader();

		while (true) {
			const { value, done } = await reader.read();
			if (done) break;

			try {
				let lines = value.split('\n');

				for (const line of lines) {
					if (line !== '' && line !== 'null') {
						console.log(line);
						let data = JSON.parse(line);
						console.log(data);

						if (data.error) {
							throw data.error;
						}
						if (data.detail) {
							throw data.detail;
						}

						if (data.status) {
						}
					} else {
						toast.success(`Deleted ${deleteModelTag}`);
					}
				}
			} catch (error) {
				console.log(error);
				toast.error(error);
			}
		}

		deleteModelTag = '';
		models.set(await getModels());
	};

	const getModels = async (url = '', type = 'all') => {
		let models = [];
		const res = await fetch(`${url ? url : $settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/tags`, {
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				...($settings.authHeader && { Authorization: $settings.authHeader }),
				...($user && { Authorization: `Bearer ${localStorage.token}` })
			}
		})
			.then(async (res) => {
				if (!res.ok) throw await res.json();
				return res.json();
			})
			.catch((error) => {
				console.log(error);
				if ('detail' in error) {
					toast.error(error.detail);
				} else {
					toast.error('Server connection failed');
				}
				return null;
			});
		console.log(res);
		models.push(...(res?.models ?? []));

		// If OpenAI API Key exists
		if (type === 'all' && $settings.OPENAI_API_KEY) {
			const API_BASE_URL = $settings.OPENAI_API_BASE_URL ?? 'https://api.openai.com/v1';

			// Validate OPENAI_API_KEY
			const openaiModelRes = await fetch(`${API_BASE_URL}/models`, {
				method: 'GET',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${$settings.OPENAI_API_KEY}`
				}
			})
				.then(async (res) => {
					if (!res.ok) throw await res.json();
					return res.json();
				})
				.catch((error) => {
					console.log(error);
					toast.error(`OpenAI: ${error?.error?.message ?? 'Network Problem'}`);
					return null;
				});

			const openAIModels = Array.isArray(openaiModelRes)
				? openaiModelRes
				: openaiModelRes?.data ?? null;

			models.push(
				...(openAIModels
					? [
							{ name: 'hr' },
							...openAIModels
								.map((model) => ({ name: model.id, external: true }))
								.filter((model) =>
									API_BASE_URL.includes('openai') ? model.name.includes('gpt') : true
								)
					  ]
					: [])
			);
		}

		return models;
	};

	onMount(() => {
		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');
		console.log(settings);

		theme = localStorage.theme ?? 'dark';
		notificationEnabled = settings.notificationEnabled ?? false;

		API_BASE_URL = settings.API_BASE_URL ?? OLLAMA_API_BASE_URL;
		system = settings.system ?? '';

		requestFormat = settings.requestFormat ?? '';

		options.seed = settings.seed ?? 0;
		options.temperature = settings.temperature ?? '';
		options.repeat_penalty = settings.repeat_penalty ?? '';
		options.top_k = settings.top_k ?? '';
		options.top_p = settings.top_p ?? '';
		options.num_ctx = settings.num_ctx ?? '';
		options = { ...options, ...settings.options };

		titleAutoGenerate = settings.titleAutoGenerate ?? true;
		speechAutoSend = settings.speechAutoSend ?? false;
		responseAutoCopy = settings.responseAutoCopy ?? false;

		gravatarEmail = settings.gravatarEmail ?? '';
		OPENAI_API_KEY = settings.OPENAI_API_KEY ?? '';
		OPENAI_API_BASE_URL = settings.OPENAI_API_BASE_URL ?? 'https://api.openai.com/v1';

		authEnabled = settings.authHeader !== undefined ? true : false;
		if (authEnabled) {
			authType = settings.authHeader.split(' ')[0];
			authContent = settings.authHeader.split(' ')[1];
		}
	});
</script>

<Modal bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-5 py-4">
			<div class=" text-lg font-medium self-center">Settings</div>
			<button
				class="self-center"
				on:click={() => {
					show = false;
				}}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
					class="w-5 h-5"
				>
					<path
						d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
					/>
				</svg>
			</button>
		</div>
		<hr class=" dark:border-gray-800" />

		<div class="flex flex-col md:flex-row w-full p-4 md:space-x-4">
			<div
				class="tabs flex flex-row overflow-x-auto space-x-1 md:space-x-0 md:space-y-1 md:flex-col flex-1 md:flex-none md:w-40 dark:text-gray-200 text-xs text-left mb-3 md:mb-0"
			>
				<button
					class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
					'general'
						? 'bg-gray-200 dark:bg-gray-700'
						: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
					on:click={() => {
						selectedTab = 'general';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M8.34 1.804A1 1 0 019.32 1h1.36a1 1 0 01.98.804l.295 1.473c.497.144.971.342 1.416.587l1.25-.834a1 1 0 011.262.125l.962.962a1 1 0 01.125 1.262l-.834 1.25c.245.445.443.919.587 1.416l1.473.294a1 1 0 01.804.98v1.361a1 1 0 01-.804.98l-1.473.295a6.95 6.95 0 01-.587 1.416l.834 1.25a1 1 0 01-.125 1.262l-.962.962a1 1 0 01-1.262.125l-1.25-.834a6.953 6.953 0 01-1.416.587l-.294 1.473a1 1 0 01-.98.804H9.32a1 1 0 01-.98-.804l-.295-1.473a6.957 6.957 0 01-1.416-.587l-1.25.834a1 1 0 01-1.262-.125l-.962-.962a1 1 0 01-.125-1.262l.834-1.25a6.957 6.957 0 01-.587-1.416l-1.473-.294A1 1 0 011 10.68V9.32a1 1 0 01.804-.98l1.473-.295c.144-.497.342-.971.587-1.416l-.834-1.25a1 1 0 01.125-1.262l.962-.962A1 1 0 015.38 3.03l1.25.834a6.957 6.957 0 011.416-.587l.294-1.473zM13 10a3 3 0 11-6 0 3 3 0 016 0z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<div class=" self-center">General</div>
				</button>

				<button
					class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
					'advanced'
						? 'bg-gray-200 dark:bg-gray-700'
						: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
					on:click={() => {
						selectedTab = 'advanced';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								d="M17 2.75a.75.75 0 00-1.5 0v5.5a.75.75 0 001.5 0v-5.5zM17 15.75a.75.75 0 00-1.5 0v1.5a.75.75 0 001.5 0v-1.5zM3.75 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5a.75.75 0 01.75-.75zM4.5 2.75a.75.75 0 00-1.5 0v5.5a.75.75 0 001.5 0v-5.5zM10 11a.75.75 0 01.75.75v5.5a.75.75 0 01-1.5 0v-5.5A.75.75 0 0110 11zM10.75 2.75a.75.75 0 00-1.5 0v1.5a.75.75 0 001.5 0v-1.5zM10 6a2 2 0 100 4 2 2 0 000-4zM3.75 10a2 2 0 100 4 2 2 0 000-4zM16.25 10a2 2 0 100 4 2 2 0 000-4z"
							/>
						</svg>
					</div>
					<div class=" self-center">Advanced</div>
				</button>

				<button
					class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
					'models'
						? 'bg-gray-200 dark:bg-gray-700'
						: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
					on:click={() => {
						selectedTab = 'models';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M10 1c3.866 0 7 1.79 7 4s-3.134 4-7 4-7-1.79-7-4 3.134-4 7-4zm5.694 8.13c.464-.264.91-.583 1.306-.952V10c0 2.21-3.134 4-7 4s-7-1.79-7-4V8.178c.396.37.842.688 1.306.953C5.838 10.006 7.854 10.5 10 10.5s4.162-.494 5.694-1.37zM3 13.179V15c0 2.21 3.134 4 7 4s7-1.79 7-4v-1.822c-.396.37-.842.688-1.306.953-1.532.875-3.548 1.369-5.694 1.369s-4.162-.494-5.694-1.37A7.009 7.009 0 013 13.179z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<div class=" self-center">Models</div>
				</button>

				<button
					class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
					'external'
						? 'bg-gray-200 dark:bg-gray-700'
						: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
					on:click={() => {
						selectedTab = 'external';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 16 16"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								d="M1 9.5A3.5 3.5 0 0 0 4.5 13H12a3 3 0 0 0 .917-5.857 2.503 2.503 0 0 0-3.198-3.019 3.5 3.5 0 0 0-6.628 2.171A3.5 3.5 0 0 0 1 9.5Z"
							/>
						</svg>
					</div>
					<div class=" self-center">External</div>
				</button>

				<button
					class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
					'addons'
						? 'bg-gray-200 dark:bg-gray-700'
						: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
					on:click={() => {
						selectedTab = 'addons';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								d="M12 4.467c0-.405.262-.75.559-1.027.276-.257.441-.584.441-.94 0-.828-.895-1.5-2-1.5s-2 .672-2 1.5c0 .362.171.694.456.953.29.265.544.6.544.994a.968.968 0 01-1.024.974 39.655 39.655 0 01-3.014-.306.75.75 0 00-.847.847c.14.993.242 1.999.306 3.014A.968.968 0 014.447 10c-.393 0-.729-.253-.994-.544C3.194 9.17 2.862 9 2.5 9 1.672 9 1 9.895 1 11s.672 2 1.5 2c.356 0 .683-.165.94-.441.276-.297.622-.559 1.027-.559a.997.997 0 011.004 1.03 39.747 39.747 0 01-.319 3.734.75.75 0 00.64.842c1.05.146 2.111.252 3.184.318A.97.97 0 0010 16.948c0-.394-.254-.73-.545-.995C9.171 15.693 9 15.362 9 15c0-.828.895-1.5 2-1.5s2 .672 2 1.5c0 .356-.165.683-.441.94-.297.276-.559.622-.559 1.027a.998.998 0 001.03 1.005c1.337-.05 2.659-.162 3.961-.337a.75.75 0 00.644-.644c.175-1.302.288-2.624.337-3.961A.998.998 0 0016.967 12c-.405 0-.75.262-1.027.559-.257.276-.584.441-.94.441-.828 0-1.5-.895-1.5-2s.672-2 1.5-2c.362 0 .694.17.953.455.265.291.601.545.995.545a.97.97 0 00.976-1.024 41.159 41.159 0 00-.318-3.184.75.75 0 00-.842-.64c-1.228.164-2.473.271-3.734.319A.997.997 0 0112 4.467z"
							/>
						</svg>
					</div>
					<div class=" self-center">Add-ons</div>
				</button>

				{#if !$config || ($config && !$config.auth)}
					<button
						class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
						'auth'
							? 'bg-gray-200 dark:bg-gray-700'
							: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
						on:click={() => {
							selectedTab = 'auth';
						}}
					>
						<div class=" self-center mr-2">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 24 24"
								fill="currentColor"
								class="w-4 h-4"
							>
								<path
									fill-rule="evenodd"
									d="M12.516 2.17a.75.75 0 00-1.032 0 11.209 11.209 0 01-7.877 3.08.75.75 0 00-.722.515A12.74 12.74 0 002.25 9.75c0 5.942 4.064 10.933 9.563 12.348a.749.749 0 00.374 0c5.499-1.415 9.563-6.406 9.563-12.348 0-1.39-.223-2.73-.635-3.985a.75.75 0 00-.722-.516l-.143.001c-2.996 0-5.717-1.17-7.734-3.08zm3.094 8.016a.75.75 0 10-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 00-1.06 1.06l2.25 2.25a.75.75 0 001.14-.094l3.75-5.25z"
									clip-rule="evenodd"
								/>
							</svg>
						</div>
						<div class=" self-center">Authentication</div>
					</button>
				{/if}

				<button
					class="px-2.5 py-2.5 min-w-fit rounded-lg flex-1 md:flex-none flex text-right transition {selectedTab ===
					'about'
						? 'bg-gray-200 dark:bg-gray-700'
						: ' hover:bg-gray-300 dark:hover:bg-gray-800'}"
					on:click={() => {
						selectedTab = 'about';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<div class=" self-center">About</div>
				</button>
			</div>
			<div class="flex-1 md:min-h-[340px]">
				{#if selectedTab === 'general'}
					<div class="flex flex-col space-y-3">
						<div>
							<div class=" mb-1 text-sm font-medium">WebUI Settings</div>

							<div class=" py-0.5 flex w-full justify-between">
								<div class=" self-center text-xs font-medium">Theme</div>

								<button
									class="p-1 px-3 text-xs flex rounded transition"
									on:click={() => {
										toggleTheme();
									}}
								>
									{#if theme === 'dark'}
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 20 20"
											fill="currentColor"
											class="w-4 h-4"
										>
											<path
												fill-rule="evenodd"
												d="M7.455 2.004a.75.75 0 01.26.77 7 7 0 009.958 7.967.75.75 0 011.067.853A8.5 8.5 0 116.647 1.921a.75.75 0 01.808.083z"
												clip-rule="evenodd"
											/>
										</svg>

										<span class="ml-2 self-center"> Dark </span>
									{:else}
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 20 20"
											fill="currentColor"
											class="w-4 h-4 self-center"
										>
											<path
												d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zM10 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zM10 7a3 3 0 100 6 3 3 0 000-6zM15.657 5.404a.75.75 0 10-1.06-1.06l-1.061 1.06a.75.75 0 001.06 1.06l1.06-1.06zM6.464 14.596a.75.75 0 10-1.06-1.06l-1.06 1.06a.75.75 0 001.06 1.06l1.06-1.06zM18 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 0118 10zM5 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 015 10zM14.596 15.657a.75.75 0 001.06-1.06l-1.06-1.061a.75.75 0 10-1.06 1.06l1.06 1.06zM5.404 6.464a.75.75 0 001.06-1.06l-1.06-1.06a.75.75 0 10-1.061 1.06l1.06 1.06z"
											/>
										</svg>
										<span class="ml-2 self-center"> Light </span>
									{/if}
								</button>
							</div>

							<div>
								<div class=" py-0.5 flex w-full justify-between">
									<div class=" self-center text-xs font-medium">Notification</div>

									<button
										class="p-1 px-3 text-xs flex rounded transition"
										on:click={() => {
											toggleNotification();
										}}
										type="button"
									>
										{#if notificationEnabled === true}
											<span class="ml-2 self-center">On</span>
										{:else}
											<span class="ml-2 self-center">Off</span>
										{/if}
									</button>
								</div>
							</div>
						</div>

						<hr class=" dark:border-gray-700" />
						<div>
							<div class=" mb-2.5 text-sm font-medium">Ollama Server URL</div>
							<div class="flex w-full">
								<div class="flex-1 mr-2">
									<input
										class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
										placeholder="Enter URL (e.g. http://localhost:11434/api)"
										bind:value={API_BASE_URL}
									/>
								</div>
								<button
									class="px-3 bg-gray-200 hover:bg-gray-300 dark:bg-gray-600 dark:hover:bg-gray-700 rounded transition"
									on:click={() => {
										checkOllamaConnection();
									}}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 20 20"
										fill="currentColor"
										class="w-4 h-4"
									>
										<path
											fill-rule="evenodd"
											d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							</div>

							<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
								Trouble accessing Ollama? <a
									class=" text-gray-500 dark:text-gray-300 font-medium"
									href="https://github.com/ollama-webui/ollama-webui#troubleshooting"
									target="_blank"
								>
									Click here for help.
								</a>
							</div>
						</div>

						<hr class=" dark:border-gray-700" />

						<div>
							<div class=" mb-2.5 text-sm font-medium">System Prompt</div>
							<textarea
								bind:value={system}
								class="w-full rounded p-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none resize-none"
								rows="4"
							/>
						</div>

						<div class="flex justify-end pt-3 text-sm font-medium">
							<button
								class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
								on:click={() => {
									saveSettings({
										API_BASE_URL: API_BASE_URL === '' ? OLLAMA_API_BASE_URL : API_BASE_URL,
										system: system !== '' ? system : undefined
									});
									show = false;
								}}
							>
								Save
							</button>
						</div>
					</div>
				{:else if selectedTab === 'advanced'}
					<div class="flex flex-col h-full justify-between text-sm">
						<div class=" space-y-3 pr-1.5 overflow-y-scroll max-h-72">
							<div class=" text-sm font-medium">Parameters</div>

							<Advanced bind:options />
							<hr class=" dark:border-gray-700" />

							<div>
								<div class=" py-1 flex w-full justify-between">
									<div class=" self-center text-sm font-medium">Request Mode</div>

									<button
										class="p-1 px-3 text-xs flex rounded transition"
										on:click={() => {
											toggleRequestFormat();
										}}
									>
										{#if requestFormat === ''}
											<span class="ml-2 self-center"> Default </span>
										{:else if requestFormat === 'json'}
											<!-- <svg
												xmlns="http://www.w3.org/2000/svg"
												viewBox="0 0 20 20"
												fill="currentColor"
												class="w-4 h-4 self-center"
											>
												<path
													d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zM10 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zM10 7a3 3 0 100 6 3 3 0 000-6zM15.657 5.404a.75.75 0 10-1.06-1.06l-1.061 1.06a.75.75 0 001.06 1.06l1.06-1.06zM6.464 14.596a.75.75 0 10-1.06-1.06l-1.06 1.06a.75.75 0 001.06 1.06l1.06-1.06zM18 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 0118 10zM5 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 015 10zM14.596 15.657a.75.75 0 001.06-1.06l-1.06-1.061a.75.75 0 10-1.06 1.06l1.06 1.06zM5.404 6.464a.75.75 0 001.06-1.06l-1.06-1.06a.75.75 0 10-1.061 1.06l1.06 1.06z"
												/>
											</svg> -->
											<span class="ml-2 self-center"> JSON </span>
										{/if}
									</button>
								</div>
							</div>
						</div>

						<div class="flex justify-end pt-3 text-sm font-medium">
							<button
								class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
								on:click={() => {
									saveSettings({
										options: {
											seed: (options.seed !== 0 ? options.seed : undefined) ?? undefined,
											stop: options.stop !== '' ? options.stop : undefined,
											temperature: options.temperature !== '' ? options.temperature : undefined,
											repeat_penalty:
												options.repeat_penalty !== '' ? options.repeat_penalty : undefined,
											repeat_last_n:
												options.repeat_last_n !== '' ? options.repeat_last_n : undefined,
											mirostat: options.mirostat !== '' ? options.mirostat : undefined,
											mirostat_eta: options.mirostat_eta !== '' ? options.mirostat_eta : undefined,
											mirostat_tau: options.mirostat_tau !== '' ? options.mirostat_tau : undefined,
											top_k: options.top_k !== '' ? options.top_k : undefined,
											top_p: options.top_p !== '' ? options.top_p : undefined,
											tfs_z: options.tfs_z !== '' ? options.tfs_z : undefined,
											num_ctx: options.num_ctx !== '' ? options.num_ctx : undefined
										}
									});
									show = false;
								}}
							>
								Save
							</button>
						</div>
					</div>
				{:else if selectedTab === 'models'}
					<div class="flex flex-col h-full justify-between text-sm">
						<div class=" space-y-3 pr-1.5 overflow-y-scroll h-80">
							<div>
								<div class=" mb-2.5 text-sm font-medium">Pull a model from Ollama.ai</div>
								<div class="flex w-full">
									<div class="flex-1 mr-2">
										<input
											class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
											placeholder="Enter model tag (e.g. mistral:7b)"
											bind:value={modelTag}
										/>
									</div>
									<button
										class="px-3 text-gray-100 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded transition"
										on:click={() => {
											pullModelHandler();
										}}
										disabled={modelTransferring}
									>
										{#if modelTransferring}
											<div class="self-center">
												<svg
													class=" w-4 h-4"
													viewBox="0 0 24 24"
													fill="currentColor"
													xmlns="http://www.w3.org/2000/svg"
													><style>
														.spinner_ajPY {
															transform-origin: center;
															animation: spinner_AtaB 0.75s infinite linear;
														}
														@keyframes spinner_AtaB {
															100% {
																transform: rotate(360deg);
															}
														}
													</style><path
														d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z"
														opacity=".25"
													/><path
														d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z"
														class="spinner_ajPY"
													/></svg
												>
											</div>
										{:else}
											<svg
												xmlns="http://www.w3.org/2000/svg"
												viewBox="0 0 16 16"
												fill="currentColor"
												class="w-4 h-4"
											>
												<path
													d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.03 6.22a.75.75 0 0 0-1.06 1.06l3.5 3.5a.75.75 0 0 0 1.06 0l3.5-3.5a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75Z"
												/>
												<path
													d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z"
												/>
											</svg>
										{/if}
									</button>
								</div>

								<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
									To access the available model names for downloading, <a
										class=" text-gray-500 dark:text-gray-300 font-medium"
										href="https://ollama.ai/library"
										target="_blank">click here.</a
									>
								</div>

								{#if pullProgress !== null}
									<div class="mt-2">
										<div class=" mb-2 text-xs">Pull Progress</div>
										<div class="w-full rounded-full dark:bg-gray-800">
											<div
												class="dark:bg-gray-600 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded-full"
												style="width: {Math.max(15, pullProgress ?? 0)}%"
											>
												{pullProgress ?? 0}%
											</div>
										</div>
										<div class="mt-1 text-xs dark:text-gray-500" style="font-size: 0.5rem;">
											{digest}
										</div>
									</div>
								{/if}
							</div>
							<hr class=" dark:border-gray-700" />

							<form
								on:submit|preventDefault={() => {
									uploadModelHandler();
								}}
							>
								<div class=" mb-2 flex w-full justify-between">
									<div class="  text-sm font-medium">Upload a GGUF model</div>

									<button
										class="p-1 px-3 text-xs flex rounded transition"
										on:click={() => {
											if (modelUploadMode === 'file') {
												modelUploadMode = 'url';
											} else {
												modelUploadMode = 'file';
											}
										}}
										type="button"
									>
										{#if modelUploadMode === 'file'}
											<span class="ml-2 self-center">File Mode</span>
										{:else}
											<span class="ml-2 self-center">URL Mode</span>
										{/if}
									</button>
								</div>

								<div class="flex w-full mb-1.5">
									<div class="flex flex-col w-full">
										{#if modelUploadMode === 'file'}
											<div
												class="flex-1 {modelInputFile && modelInputFile.length > 0 ? 'mr-2' : ''}"
											>
												<input
													id="model-upload-input"
													type="file"
													bind:files={modelInputFile}
													on:change={() => {
														console.log(modelInputFile);
													}}
													accept=".gguf"
													required
													hidden
												/>

												<button
													type="button"
													class="w-full rounded text-left py-2 px-4 dark:text-gray-300 dark:bg-gray-800"
													on:click={() => {
														document.getElementById('model-upload-input').click();
													}}
												>
													{#if modelInputFile && modelInputFile.length > 0}
														{modelInputFile[0].name}
													{:else}
														Click here to select
													{/if}
												</button>
											</div>
										{:else}
											<div class="flex-1 {modelFileUrl !== '' ? 'mr-2' : ''}">
												<input
													class="w-full rounded text-left py-2 px-4 dark:text-gray-300 dark:bg-gray-800 outline-none {modelFileUrl !==
													''
														? 'mr-2'
														: ''}"
													type="url"
													required
													bind:value={modelFileUrl}
													placeholder="Type HuggingFace Resolve (Download) URL"
												/>
											</div>
										{/if}
									</div>

									{#if (modelUploadMode === 'file' && modelInputFile && modelInputFile.length > 0) || (modelUploadMode === 'url' && modelFileUrl !== '')}
										<button
											class="px-3 text-gray-100 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded transition"
											type="submit"
											disabled={modelTransferring}
										>
											{#if modelTransferring}
												<div class="self-center">
													<svg
														class=" w-4 h-4"
														viewBox="0 0 24 24"
														fill="currentColor"
														xmlns="http://www.w3.org/2000/svg"
														><style>
															.spinner_ajPY {
																transform-origin: center;
																animation: spinner_AtaB 0.75s infinite linear;
															}
															@keyframes spinner_AtaB {
																100% {
																	transform: rotate(360deg);
																}
															}
														</style><path
															d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z"
															opacity=".25"
														/><path
															d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z"
															class="spinner_ajPY"
														/></svg
													>
												</div>
											{:else}
												<svg
													xmlns="http://www.w3.org/2000/svg"
													viewBox="0 0 16 16"
													fill="currentColor"
													class="w-4 h-4"
												>
													<path
														d="M7.25 10.25a.75.75 0 0 0 1.5 0V4.56l2.22 2.22a.75.75 0 1 0 1.06-1.06l-3.5-3.5a.75.75 0 0 0-1.06 0l-3.5 3.5a.75.75 0 0 0 1.06 1.06l2.22-2.22v5.69Z"
													/>
													<path
														d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z"
													/>
												</svg>
											{/if}
										</button>
									{/if}
								</div>

								{#if (modelUploadMode === 'file' && modelInputFile && modelInputFile.length > 0) || (modelUploadMode === 'url' && modelFileUrl !== '')}
									<div>
										<div>
											<div class=" my-2.5 text-sm font-medium">Modelfile Content</div>
											<textarea
												bind:value={modelFileContent}
												class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none resize-none"
												rows="6"
											/>
										</div>
									</div>
								{/if}
								<div class=" mt-1 text-xs text-gray-400 dark:text-gray-500">
									To access the GGUF models available for downloading, <a
										class=" text-gray-500 dark:text-gray-300 font-medium"
										href="https://huggingface.co/models?search=gguf"
										target="_blank">click here.</a
									>
								</div>

								{#if uploadProgress !== null}
									<div class="mt-2">
										<div class=" mb-2 text-xs">Upload Progress</div>
										<div class="w-full rounded-full dark:bg-gray-800">
											<div
												class="dark:bg-gray-600 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded-full"
												style="width: {Math.max(15, uploadProgress ?? 0)}%"
											>
												{uploadProgress ?? 0}%
											</div>
										</div>
										<div class="mt-1 text-xs dark:text-gray-500" style="font-size: 0.5rem;">
											{modelFileDigest}
										</div>
									</div>
								{/if}
							</form>
							<hr class=" dark:border-gray-700" />

							<div>
								<div class=" mb-2.5 text-sm font-medium">Delete a model</div>
								<div class="flex w-full">
									<div class="flex-1 mr-2">
										<select
											class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
											bind:value={deleteModelTag}
											placeholder="Select a model"
										>
											{#if !deleteModelTag}
												<option value="" disabled selected>Select a model</option>
											{/if}
											{#each $models.filter((m) => m.size != null) as model}
												<option value={model.name} class="bg-gray-100 dark:bg-gray-700"
													>{model.name +
														' (' +
														(model.size / 1024 ** 3).toFixed(1) +
														' GB)'}</option
												>
											{/each}
										</select>
									</div>
									<button
										class="px-3 bg-red-700 hover:bg-red-800 text-gray-100 rounded transition"
										on:click={() => {
											deleteModelHandler();
										}}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 16 16"
											fill="currentColor"
											class="w-4 h-4"
										>
											<path
												fill-rule="evenodd"
												d="M5 3.25V4H2.75a.75.75 0 0 0 0 1.5h.3l.815 8.15A1.5 1.5 0 0 0 5.357 15h5.285a1.5 1.5 0 0 0 1.493-1.35l.815-8.15h.3a.75.75 0 0 0 0-1.5H11v-.75A2.25 2.25 0 0 0 8.75 1h-1.5A2.25 2.25 0 0 0 5 3.25Zm2.25-.75a.75.75 0 0 0-.75.75V4h3v-.75a.75.75 0 0 0-.75-.75h-1.5ZM6.05 6a.75.75 0 0 1 .787.713l.275 5.5a.75.75 0 0 1-1.498.075l-.275-5.5A.75.75 0 0 1 6.05 6Zm3.9 0a.75.75 0 0 1 .712.787l-.275 5.5a.75.75 0 0 1-1.498-.075l.275-5.5a.75.75 0 0 1 .786-.711Z"
												clip-rule="evenodd"
											/>
										</svg>
									</button>
								</div>
							</div>
						</div>
					</div>
				{:else if selectedTab === 'external'}
					<form
						class="flex flex-col h-full justify-between space-y-3 text-sm"
						on:submit|preventDefault={() => {
							saveSettings({
								OPENAI_API_KEY: OPENAI_API_KEY !== '' ? OPENAI_API_KEY : undefined,
								OPENAI_API_BASE_URL: OPENAI_API_BASE_URL !== '' ? OPENAI_API_BASE_URL : undefined
							});
							show = false;
						}}
					>
						<div class=" space-y-3">
							<div>
								<div class=" mb-2.5 text-sm font-medium">OpenAI API Key</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
											placeholder="Enter OpenAI API Key"
											bind:value={OPENAI_API_KEY}
											autocomplete="off"
										/>
									</div>
								</div>
								<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
									Adds optional support for online models.
								</div>
							</div>

							<hr class=" dark:border-gray-700" />

							<div>
								<div class=" mb-2.5 text-sm font-medium">OpenAI API Base URL</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
											placeholder="Enter OpenAI API Key"
											bind:value={OPENAI_API_BASE_URL}
											autocomplete="off"
										/>
									</div>
								</div>
								<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
									WebUI will make requests to <span class=" text-gray-200"
										>'{OPENAI_API_BASE_URL}/chat'</span
									>
								</div>
							</div>
						</div>

						<div class="flex justify-end pt-3 text-sm font-medium">
							<button
								class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
								type="submit"
							>
								Save
							</button>
						</div>
					</form>
				{:else if selectedTab === 'addons'}
					<form
						class="flex flex-col h-full justify-between space-y-3 text-sm"
						on:submit|preventDefault={() => {
							saveSettings({
								gravatarEmail: gravatarEmail !== '' ? gravatarEmail : undefined,
								gravatarUrl: gravatarEmail !== '' ? getGravatarURL(gravatarEmail) : undefined
							});
							show = false;
						}}
					>
						<div class=" space-y-3">
							<div>
								<div class=" mb-1 text-sm font-medium">WebUI Add-ons</div>

								<div>
									<div class=" py-0.5 flex w-full justify-between">
										<div class=" self-center text-xs font-medium">Title Auto Generation</div>

										<button
											class="p-1 px-3 text-xs flex rounded transition"
											on:click={() => {
												toggleTitleAutoGenerate();
											}}
											type="button"
										>
											{#if titleAutoGenerate === true}
												<span class="ml-2 self-center">On</span>
											{:else}
												<span class="ml-2 self-center">Off</span>
											{/if}
										</button>
									</div>
								</div>

								<div>
									<div class=" py-0.5 flex w-full justify-between">
										<div class=" self-center text-xs font-medium">Voice Input Auto-Send</div>

										<button
											class="p-1 px-3 text-xs flex rounded transition"
											on:click={() => {
												toggleSpeechAutoSend();
											}}
											type="button"
										>
											{#if speechAutoSend === true}
												<span class="ml-2 self-center">On</span>
											{:else}
												<span class="ml-2 self-center">Off</span>
											{/if}
										</button>
									</div>
								</div>

								<div>
									<div class=" py-0.5 flex w-full justify-between">
										<div class=" self-center text-xs font-medium">
											Response AutoCopy to Clipboard
										</div>

										<button
											class="p-1 px-3 text-xs flex rounded transition"
											on:click={() => {
												toggleResponseAutoCopy();
											}}
											type="button"
										>
											{#if responseAutoCopy === true}
												<span class="ml-2 self-center">On</span>
											{:else}
												<span class="ml-2 self-center">Off</span>
											{/if}
										</button>
									</div>
								</div>
							</div>

							<hr class=" dark:border-gray-700" />
							<div>
								<div class=" mb-2.5 text-sm font-medium">
									Gravatar Email <span class=" text-gray-400 text-sm">(optional)</span>
								</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
											placeholder="Enter Your Email"
											bind:value={gravatarEmail}
											autocomplete="off"
											type="email"
										/>
									</div>
								</div>
								<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
									Changes user profile image to match your <a
										class=" text-gray-500 dark:text-gray-300 font-medium"
										href="https://gravatar.com/"
										target="_blank">Gravatar.</a
									>
								</div>
							</div>
						</div>

						<div class="flex justify-end pt-3 text-sm font-medium">
							<button
								class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
								type="submit"
							>
								Save
							</button>
						</div>
					</form>
				{:else if selectedTab === 'auth'}
					<form
						class="flex flex-col h-full justify-between space-y-3 text-sm"
						on:submit|preventDefault={() => {
							console.log('auth save');
							saveSettings({
								authHeader: authEnabled ? `${authType} ${authContent}` : undefined
							});
							show = false;
						}}
					>
						<div class=" space-y-3">
							<div>
								<div class=" py-1 flex w-full justify-between">
									<div class=" self-center text-sm font-medium">Authorization Header</div>

									<button
										class="p-1 px-3 text-xs flex rounded transition"
										type="button"
										on:click={() => {
											toggleAuthHeader();
										}}
									>
										{#if authEnabled === true}
											<svg
												xmlns="http://www.w3.org/2000/svg"
												viewBox="0 0 24 24"
												fill="currentColor"
												class="w-4 h-4"
											>
												<path
													fill-rule="evenodd"
													d="M12 1.5a5.25 5.25 0 00-5.25 5.25v3a3 3 0 00-3 3v6.75a3 3 0 003 3h10.5a3 3 0 003-3v-6.75a3 3 0 00-3-3v-3c0-2.9-2.35-5.25-5.25-5.25zm3.75 8.25v-3a3.75 3.75 0 10-7.5 0v3h7.5z"
													clip-rule="evenodd"
												/>
											</svg>

											<span class="ml-2 self-center"> On </span>
										{:else}
											<svg
												xmlns="http://www.w3.org/2000/svg"
												viewBox="0 0 24 24"
												fill="currentColor"
												class="w-4 h-4"
											>
												<path
													d="M18 1.5c2.9 0 5.25 2.35 5.25 5.25v3.75a.75.75 0 01-1.5 0V6.75a3.75 3.75 0 10-7.5 0v3a3 3 0 013 3v6.75a3 3 0 01-3 3H3.75a3 3 0 01-3-3v-6.75a3 3 0 013-3h9v-3c0-2.9 2.35-5.25 5.25-5.25z"
												/>
											</svg>

											<span class="ml-2 self-center">Off</span>
										{/if}
									</button>
								</div>
							</div>

							{#if authEnabled}
								<hr class=" dark:border-gray-700" />

								<div class="mt-2">
									<div class=" py-1 flex w-full space-x-2">
										<button
											class=" py-1 font-semibold flex rounded transition"
											on:click={() => {
												authType = authType === 'Basic' ? 'Bearer' : 'Basic';
											}}
											type="button"
										>
											{#if authType === 'Basic'}
												<span class="self-center mr-2">Basic</span>
											{:else if authType === 'Bearer'}
												<span class="self-center mr-2">Bearer</span>
											{/if}
										</button>

										<div class="flex-1">
											<input
												class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
												placeholder="Enter Authorization Header Content"
												bind:value={authContent}
											/>
										</div>
									</div>
									<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
										Toggle between <span class=" text-gray-500 dark:text-gray-300 font-medium"
											>'Basic'</span
										>
										and <span class=" text-gray-500 dark:text-gray-300 font-medium">'Bearer'</span> by
										clicking on the label next to the input.
									</div>
								</div>

								<hr class=" dark:border-gray-700" />

								<div>
									<div class=" mb-2.5 text-sm font-medium">Preview Authorization Header</div>
									<textarea
										value={JSON.stringify({
											Authorization: `${authType} ${authContent}`
										})}
										class="w-full rounded p-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none resize-none"
										rows="2"
										disabled
									/>
								</div>
							{/if}
						</div>

						<div class="flex justify-end pt-3 text-sm font-medium">
							<button
								class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
								type="submit"
							>
								Save
							</button>
						</div>
					</form>
				{:else if selectedTab === 'about'}
					<div class="flex flex-col h-full justify-between space-y-3 text-sm mb-6">
						<div class=" space-y-3">
							<div>
								<div class=" mb-2.5 text-sm font-medium">Ollama Web UI Version</div>
								<div class="flex w-full">
									<div class="flex-1 text-xs text-gray-700 dark:text-gray-200">
										{$config && $config.version ? $config.version : WEB_UI_VERSION}
									</div>
								</div>
							</div>

							<hr class=" dark:border-gray-700" />

							<div>
								<div class=" mb-2.5 text-sm font-medium">Ollama Version</div>
								<div class="flex w-full">
									<div class="flex-1 text-xs text-gray-700 dark:text-gray-200">
										{$info?.ollama?.version ?? 'N/A'}
									</div>
								</div>
							</div>

							<hr class=" dark:border-gray-700" />

							<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
								Created by <a
									class=" text-gray-500 dark:text-gray-300 font-medium"
									href="https://github.com/tjbck"
									target="_blank">Timothy J. Baek</a
								>
							</div>

							<div>
								<a href="https://github.com/ollama-webui/ollama-webui">
									<img
										alt="Github Repo"
										src="https://img.shields.io/github/stars/ollama-webui/ollama-webui?style=social&label=Star us on Github"
									/>
								</a>
							</div>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
</Modal>

<style>
	input::-webkit-outer-spin-button,
	input::-webkit-inner-spin-button {
		/* display: none; <- Crashes Chrome on hover */
		-webkit-appearance: none;
		margin: 0; /* <-- Apparently some margin are still there even though it's hidden */
	}

	.tabs::-webkit-scrollbar {
		display: none; /* for Chrome, Safari and Opera */
	}

	.tabs {
		-ms-overflow-style: none; /* IE and Edge */
		scrollbar-width: none; /* Firefox */
	}

	input[type='number'] {
		-moz-appearance: textfield; /* Firefox */
	}
</style>
