<script lang="ts">
	import toast from 'svelte-french-toast';
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';

	import { getOllamaModels, getOllamaVersion } from '$lib/apis/ollama';
	import { getOpenAIModels } from '$lib/apis/openai';

	import { user, showSettings, settings, models, modelfiles } from '$lib/stores';
	import { OLLAMA_API_BASE_URL, REQUIRED_OLLAMA_VERSION, WEBUI_API_BASE_URL } from '$lib/constants';

	import SettingsModal from '$lib/components/chat/SettingsModal.svelte';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';
	import { checkVersion } from '$lib/utils';

	let ollamaVersion = '';
	let loaded = false;

	const getModels = async () => {
		let models = [];
		models.push(
			...(await getOllamaModels(
				$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL,
				localStorage.token
			).catch((error) => {
				toast.error(error);
				return [];
			}))
		);
		// If OpenAI API Key exists
		if ($settings.OPENAI_API_KEY) {
			const openAIModels = await getOpenAIModels(
				$settings.OPENAI_API_BASE_URL ?? 'https://api.openai.com/v1',
				$settings.OPENAI_API_KEY
			).catch((error) => {
				console.log(error);
				toast.error(error);
				return null;
			});

			models.push(...(openAIModels ? [{ name: 'hr' }, ...openAIModels] : []));
		}
		return models;
	};

	const setOllamaVersion = async (version: string = '') => {
		if (version === '') {
			version = await getOllamaVersion(
				$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL,
				localStorage.token
			).catch((error) => {
				return '';
			});
		}

		ollamaVersion = version;

		console.log(ollamaVersion);
		if (checkVersion(REQUIRED_OLLAMA_VERSION, ollamaVersion)) {
			toast.error(`Ollama Version: ${ollamaVersion !== '' ? ollamaVersion : 'Not Detected'}`);
		}
	};

	onMount(async () => {
		if ($user === undefined) {
			await goto('/auth');
		} else if (['user', 'admin'].includes($user.role)) {
			await settings.set(JSON.parse(localStorage.getItem('settings') ?? '{}'));
			await models.set(await getModels());

			await modelfiles.set(JSON.parse(localStorage.getItem('modelfiles') ?? '[]'));
			modelfiles.subscribe(async () => {
				// should fetch models
			});

			await setOllamaVersion();
			await tick();
		}

		loaded = true;
	});
</script>

{#if loaded}
	<div class="app relative">
		{#if !['user', 'admin'].includes($user.role)}
			<div class="absolute w-full h-full flex z-50">
				<div
					class="absolute rounded-xl w-full h-full backdrop-blur bg-gray-900/60 flex justify-center"
				>
					<div class="m-auto pb-44 flex flex-col justify-center">
						<div class="max-w-md">
							<div class="text-center dark:text-white text-2xl font-medium z-50">
								Account Activation Pending<br /> Contact Admin for WebUI Access
							</div>

							<div class=" mt-4 text-center text-sm dark:text-gray-200 w-full">
								Your account status is currently pending activation. To access the WebUI, please
								reach out to the administrator. Admins can manage user statuses from the Admin
								Panel.
							</div>

							<div class=" mt-6 mx-auto relative group w-fit">
								<button
									class="relative z-20 flex px-5 py-2 rounded-full bg-gray-100 hover:bg-gray-200 transition font-medium text-sm"
									on:click={async () => {
										location.href = '/';
									}}
								>
									Check Again
								</button>

								<button
									class="text-xs text-center w-full mt-2 text-gray-400 underline"
									on:click={async () => {
										localStorage.removeItem('token');
										location.href = '/auth';
									}}>Sign Out</button
								>
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if checkVersion(REQUIRED_OLLAMA_VERSION, ollamaVersion ?? '0')}
			<div class="absolute w-full h-full flex z-50">
				<div
					class="absolute rounded-xl w-full h-full backdrop-blur bg-gray-900/60 flex justify-center"
				>
					<div class="m-auto pb-44 flex flex-col justify-center">
						<div class="max-w-md">
							<div class="text-center dark:text-white text-2xl font-medium z-50">
								Connection Issue or Update Needed
							</div>

							<div class=" mt-4 text-center text-sm dark:text-gray-200 w-full">
								Oops! It seems like your Ollama needs a little attention. <br
									class=" hidden sm:flex"
								/>We've detected either a connection hiccup or observed that you're using an older
								version. Ensure you're on the latest Ollama version
								<br class=" hidden sm:flex" />(version
								<span class=" dark:text-white font-medium">{REQUIRED_OLLAMA_VERSION} or higher</span
								>) or check your connection.
							</div>

							<div class=" mt-6 mx-auto relative group w-fit">
								<button
									class="relative z-20 flex px-5 py-2 rounded-full bg-gray-100 hover:bg-gray-200 transition font-medium text-sm"
									on:click={async () => {
										await setOllamaVersion();
									}}
								>
									Check Again
								</button>

								<button
									class="text-xs text-center w-full mt-2 text-gray-400 underline"
									on:click={async () => {
										await setOllamaVersion(REQUIRED_OLLAMA_VERSION);
									}}>Close</button
								>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}

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
