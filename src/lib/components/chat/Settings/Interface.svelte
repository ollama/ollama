<script lang="ts">
	import { getBackendConfig } from '$lib/apis';
	import { setDefaultPromptSuggestions } from '$lib/apis/configs';
	import { config, user } from '$lib/stores';
	import { createEventDispatcher, onMount } from 'svelte';
	const dispatch = createEventDispatcher();

	// Interface
	let promptSuggestions = [];

	const updateInterfaceHandler = async () => {
		promptSuggestions = await setDefaultPromptSuggestions(localStorage.token, promptSuggestions);
		await config.set(await getBackendConfig());
	};

	onMount(async () => {
		if ($user.role === 'admin') {
			promptSuggestions = $config?.default_prompt_suggestions;
		}
	});
</script>

<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={() => {
		updateInterfaceHandler();
		dispatch('save');
	}}
>
	<div class=" space-y-3 pr-1.5 overflow-y-scroll max-h-80">
		<div class="flex w-full justify-between mb-2">
			<div class=" self-center text-sm font-semibold">Default Prompt Suggestions</div>

			<button
				class="p-1 px-3 text-xs flex rounded transition"
				type="button"
				on:click={() => {
					if (promptSuggestions.length === 0 || promptSuggestions.at(-1).content !== '') {
						promptSuggestions = [...promptSuggestions, { content: '', title: ['', ''] }];
					}
				}}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
					class="w-4 h-4"
				>
					<path
						d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z"
					/>
				</svg>
			</button>
		</div>
		<div class="flex flex-col space-y-1">
			{#each promptSuggestions as prompt, promptIdx}
				<div class=" flex border dark:border-gray-600 rounded-lg">
					<div class="flex flex-col flex-1">
						<div class="flex border-b dark:border-gray-600 w-full">
							<input
								class="px-3 py-1.5 text-xs w-full bg-transparent outline-none border-r dark:border-gray-600"
								placeholder="Title (e.g. Tell me a fun fact)"
								bind:value={prompt.title[0]}
							/>

							<input
								class="px-3 py-1.5 text-xs w-full bg-transparent outline-none border-r dark:border-gray-600"
								placeholder="Subtitle (e.g. about the Roman Empire)"
								bind:value={prompt.title[1]}
							/>
						</div>

						<input
							class="px-3 py-1.5 text-xs w-full bg-transparent outline-none border-r dark:border-gray-600"
							placeholder="Prompt (e.g. Tell me a fun fact about the Roman Empire)"
							bind:value={prompt.content}
						/>
					</div>

					<button
						class="px-2"
						type="button"
						on:click={() => {
							promptSuggestions.splice(promptIdx, 1);
							promptSuggestions = promptSuggestions;
						}}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
							/>
						</svg>
					</button>
				</div>
			{/each}
		</div>

		{#if promptSuggestions.length > 0}
			<div class="text-xs text-left w-full mt-2">
				Adjusting these settings will apply changes universally to all users.
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
