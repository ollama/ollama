<script lang="ts">
	import { setDefaultModels } from '$lib/apis/configs';
	import { models, showSettings, settings, user } from '$lib/stores';
	import { onMount, tick } from 'svelte';
	import toast from 'svelte-french-toast';

	export let selectedModels = [''];
	export let disabled = false;

	const saveDefaultModel = async () => {
		const hasEmptyModel = selectedModels.filter((it) => it === '');
		if (hasEmptyModel.length) {
			toast.error('Choose a model before saving...');
			return;
		}
		settings.set({ ...$settings, models: selectedModels });
		localStorage.setItem('settings', JSON.stringify($settings));

		if ($user.role === 'admin') {
			console.log('setting default models globally');
			await setDefaultModels(localStorage.token, selectedModels.join(','));
		}
		toast.success('Default model updated');
	};

	$: if (selectedModels.length > 0 && $models.length > 0) {
		selectedModels = selectedModels.map((model) =>
			$models.map((m) => m.name).includes(model) ? model : ''
		);
	}
</script>

<div class="flex flex-col my-2">
	{#each selectedModels as selectedModel, selectedModelIdx}
		<div class="flex">
			<select
				id="models"
				class="outline-none bg-transparent text-lg font-semibold rounded-lg block w-full placeholder-gray-400"
				bind:value={selectedModel}
				{disabled}
			>
				<option class=" text-gray-700" value="" selected disabled>Select a model</option>

				{#each $models as model}
					{#if model.name === 'hr'}
						<hr />
					{:else}
						<option value={model.name} class="text-gray-700 text-lg"
							>{model.name +
								`${model.size ? ` (${(model.size / 1024 ** 3).toFixed(1)}GB)` : ''}`}</option
						>
					{/if}
				{/each}
			</select>

			{#if selectedModelIdx === 0}
				<button
					class="  self-center {selectedModelIdx === 0
						? 'mr-3'
						: 'mr-7'} disabled:text-gray-600 disabled:hover:text-gray-600"
					disabled={selectedModels.length === 3 || disabled}
					on:click={() => {
						if (selectedModels.length < 3) {
							selectedModels = [...selectedModels, ''];
						}
					}}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						stroke-width="1.5"
						stroke="currentColor"
						class="w-4 h-4"
					>
						<path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m6-6H6" />
					</svg>
				</button>
			{:else}
				<button
					class="  self-center disabled:text-gray-600 disabled:hover:text-gray-600 {selectedModelIdx ===
					0
						? 'mr-3'
						: 'mr-7'}"
					{disabled}
					on:click={() => {
						selectedModels.splice(selectedModelIdx, 1);
						selectedModels = selectedModels;
					}}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						stroke-width="1.5"
						stroke="currentColor"
						class="w-4 h-4"
					>
						<path stroke-linecap="round" stroke-linejoin="round" d="M19.5 12h-15" />
					</svg>
				</button>
			{/if}

			{#if selectedModelIdx === 0}
				<button
					class=" self-center dark:hover:text-gray-300"
					id="open-settings-button"
					on:click={async () => {
						await showSettings.set(!$showSettings);
					}}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						stroke-width="1.5"
						stroke="currentColor"
						class="w-4 h-4"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M10.343 3.94c.09-.542.56-.94 1.11-.94h1.093c.55 0 1.02.398 1.11.94l.149.894c.07.424.384.764.78.93.398.164.855.142 1.205-.108l.737-.527a1.125 1.125 0 011.45.12l.773.774c.39.389.44 1.002.12 1.45l-.527.737c-.25.35-.272.806-.107 1.204.165.397.505.71.93.78l.893.15c.543.09.94.56.94 1.109v1.094c0 .55-.397 1.02-.94 1.11l-.893.149c-.425.07-.765.383-.93.78-.165.398-.143.854.107 1.204l.527.738c.32.447.269 1.06-.12 1.45l-.774.773a1.125 1.125 0 01-1.449.12l-.738-.527c-.35-.25-.806-.272-1.203-.107-.397.165-.71.505-.781.929l-.149.894c-.09.542-.56.94-1.11.94h-1.094c-.55 0-1.019-.398-1.11-.94l-.148-.894c-.071-.424-.384-.764-.781-.93-.398-.164-.854-.142-1.204.108l-.738.527c-.447.32-1.06.269-1.45-.12l-.773-.774a1.125 1.125 0 01-.12-1.45l.527-.737c.25-.35.273-.806.108-1.204-.165-.397-.505-.71-.93-.78l-.894-.15c-.542-.09-.94-.56-.94-1.109v-1.094c0-.55.398-1.02.94-1.11l.894-.149c.424-.07.765-.383.93-.78.165-.398.143-.854-.107-1.204l-.527-.738a1.125 1.125 0 01.12-1.45l.773-.773a1.125 1.125 0 011.45-.12l.737.527c.35.25.807.272 1.204.107.397-.165.71-.505.78-.929l.15-.894z"
						/>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
						/>
					</svg>
				</button>
			{/if}
		</div>
	{/each}
</div>

<div class="text-left mt-1.5 text-xs text-gray-500">
	<button on:click={saveDefaultModel}> Set as default</button>
</div>
