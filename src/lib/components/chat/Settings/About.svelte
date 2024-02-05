<script lang="ts">
	import { getOllamaVersion } from '$lib/apis/ollama';
	import { WEB_UI_VERSION } from '$lib/constants';
	import { config } from '$lib/stores';
	import { onMount } from 'svelte';

	let ollamaVersion = '';
	onMount(async () => {
		ollamaVersion = await getOllamaVersion(localStorage.token).catch((error) => {
			return '';
		});
	});
</script>

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
					{ollamaVersion ?? 'N/A'}
				</div>
			</div>
		</div>

		<hr class=" dark:border-gray-700" />

		<div class="flex space-x-1">
			<a href="https://discord.gg/5rJgQTnV4s" target="_blank">
				<img
					alt="Discord"
					src="https://img.shields.io/badge/Discord-Ollama_Web_UI-blue?logo=discord&logoColor=white"
				/>
			</a>

			<a href="https://github.com/ollama-webui/ollama-webui" target="_blank">
				<img
					alt="Github Repo"
					src="https://img.shields.io/github/stars/ollama-webui/ollama-webui?style=social&label=Star us on Github"
				/>
			</a>
		</div>

		<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
			Created by <a
				class=" text-gray-500 dark:text-gray-300 font-medium"
				href="https://github.com/tjbck"
				target="_blank">Timothy J. Baek</a
			>
		</div>
	</div>
</div>
