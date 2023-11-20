<script lang="ts">
	import { openDB, deleteDB } from 'idb';
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';

	import { config, user, showSettings, settings, models, db } from '$lib/stores';

	import SettingsModal from '$lib/components/chat/SettingsModal.svelte';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';
	import toast from 'svelte-french-toast';
	import { OLLAMA_API_BASE_URL } from '$lib/constants';

	let loaded = false;

	const getModels = async () => {
		let models = [];
		const res = await fetch(`${$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/tags`, {
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
		if ($settings.OPENAI_API_KEY) {
			// Validate OPENAI_API_KEY
			const openaiModelRes = await fetch(`https://api.openai.com/v1/models`, {
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

			const openAIModels = openaiModelRes?.data ?? null;

			models.push(
				...(openAIModels
					? [
							{ name: 'hr' },
							...openAIModels
								.map((model) => ({ name: model.id, label: 'OpenAI' }))
								.filter((model) => model.name.includes('gpt'))
					  ]
					: [])
			);
		}

		return models;
	};

	const getDB = async () => {
		return await openDB('Chats', 1, {
			upgrade(db) {
				const store = db.createObjectStore('chats', {
					keyPath: 'id',
					autoIncrement: true
				});
				store.createIndex('timestamp', 'timestamp');
			}
		});
	};

	onMount(async () => {
		if ($config && $config.auth && $user === undefined) {
			await goto('/auth');
		}

		let _models = await getModels();
		await models.set(_models);
		let _db = await getDB();
		await db.set(_db);

		await tick();
		loaded = true;
	});
</script>

{#if loaded}
	<div class="app">
		<div
			class=" text-gray-700 dark:text-gray-100 bg-white dark:bg-gray-800 min-h-screen overflow-auto flex flex-row"
		>
			<Sidebar />

			<SettingsModal bind:show={$showSettings} />

			<slot />
		</div>
	</div>
{/if}

<style>
	.loading {
		display: inline-block;
		clip-path: inset(0 1ch 0 0);
		animation: l 1s steps(3) infinite;
		letter-spacing: -0.5px;
	}

	@keyframes l {
		to {
			clip-path: inset(0 -1ch 0 0);
		}
	}

	pre[class*='language-'] {
		position: relative;
		overflow: auto;

		/* make space  */
		margin: 5px 0;
		padding: 1.75rem 0 1.75rem 1rem;
		border-radius: 10px;
	}

	pre[class*='language-'] button {
		position: absolute;
		top: 5px;
		right: 5px;

		font-size: 0.9rem;
		padding: 0.15rem;
		background-color: #828282;

		border: ridge 1px #7b7b7c;
		border-radius: 5px;
		text-shadow: #c4c4c4 0 0 2px;
	}

	pre[class*='language-'] button:hover {
		cursor: pointer;
		background-color: #bcbabb;
	}
</style>
