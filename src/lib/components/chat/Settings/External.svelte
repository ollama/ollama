<script lang="ts">
	import { getOpenAIKey, getOpenAIUrl, updateOpenAIKey, updateOpenAIUrl } from '$lib/apis/openai';
	import { models, user } from '$lib/stores';
	import { createEventDispatcher, onMount } from 'svelte';
	const dispatch = createEventDispatcher();

	export let getModels: Function;

	// External
	let OPENAI_API_KEY = '';
	let OPENAI_API_BASE_URL = '';

	const updateOpenAIHandler = async () => {
		OPENAI_API_BASE_URL = await updateOpenAIUrl(localStorage.token, OPENAI_API_BASE_URL);
		OPENAI_API_KEY = await updateOpenAIKey(localStorage.token, OPENAI_API_KEY);

		await models.set(await getModels());
	};

	onMount(async () => {
		if ($user.role === 'admin') {
			OPENAI_API_BASE_URL = await getOpenAIUrl(localStorage.token);
			OPENAI_API_KEY = await getOpenAIKey(localStorage.token);
		}
	});
</script>

<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={() => {
		updateOpenAIHandler();
		dispatch('save');

		// saveSettings({
		// 	OPENAI_API_KEY: OPENAI_API_KEY !== '' ? OPENAI_API_KEY : undefined,
		// 	OPENAI_API_BASE_URL: OPENAI_API_BASE_URL !== '' ? OPENAI_API_BASE_URL : undefined
		// });
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
				WebUI will make requests to <span class=" text-gray-200">'{OPENAI_API_BASE_URL}/chat'</span>
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
