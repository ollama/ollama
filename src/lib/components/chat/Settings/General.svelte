<script lang="ts">
	import toast from 'svelte-french-toast';
	import { createEventDispatcher, onMount } from 'svelte';
	const dispatch = createEventDispatcher();

	import { getOllamaAPIUrl, updateOllamaAPIUrl } from '$lib/apis/ollama';
	import { models, user } from '$lib/stores';

	export let saveSettings: Function;
	export let getModels: Function;

	// General
	let API_BASE_URL = '';
	let themes = ['dark', 'light', 'rose-pine dark', 'rose-pine-dawn light'];
	let theme = 'dark';
	let notificationEnabled = false;
	let system = '';

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

	const updateOllamaAPIUrlHandler = async () => {
		API_BASE_URL = await updateOllamaAPIUrl(localStorage.token, API_BASE_URL);
		const _models = await getModels('ollama');

		if (_models.length > 0) {
			toast.success('Server connection verified');
			await models.set(_models);
		}
	};

	onMount(async () => {
		if ($user.role === 'admin') {
			API_BASE_URL = await getOllamaAPIUrl(localStorage.token);
		}

		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');

		theme = localStorage.theme ?? 'dark';
		notificationEnabled = settings.notificationEnabled ?? false;
		system = settings.system ?? '';
	});
</script>

<div class="flex flex-col space-y-3">
	<div>
		<div class=" mb-1 text-sm font-medium">WebUI Settings</div>

		<div class=" py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">Theme</div>
			<div class="flex items-center relative">
				<div class=" absolute right-16">
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
					{:else if theme === 'light'}
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
					{/if}
				</div>

				<select
					class="w-fit pr-8 rounded py-2 px-2 text-xs bg-transparent outline-none text-right"
					bind:value={theme}
					placeholder="Select a theme"
					on:change={(e) => {
						localStorage.theme = theme;

						themes
							.filter((e) => e !== theme)
							.forEach((e) => {
								e.split(' ').forEach((e) => {
									document.documentElement.classList.remove(e);
								});
							});

						theme.split(' ').forEach((e) => {
							document.documentElement.classList.add(e);
						});

						console.log(theme);
					}}
				>
					<option value="dark">Dark</option>
					<option value="light">Light</option>
					<option value="rose-pine dark">Rosé Pine</option>
					<option value="rose-pine-dawn light">Rosé Pine Dawn</option>
				</select>
			</div>
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

	{#if $user.role === 'admin'}
		<hr class=" dark:border-gray-700" />
		<div>
			<div class=" mb-2.5 text-sm font-medium">Ollama API URL</div>
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
						updateOllamaAPIUrlHandler();
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
				Trouble accessing Ollama?
				<a
					class=" text-gray-300 font-medium"
					href="https://github.com/ollama-webui/ollama-webui#troubleshooting"
					target="_blank"
				>
					Click here for help.
				</a>
			</div>
		</div>
	{/if}

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
					system: system !== '' ? system : undefined
				});
				dispatch('save');
			}}
		>
			Save
		</button>
	</div>
</div>
