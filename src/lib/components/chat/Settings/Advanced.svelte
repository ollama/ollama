<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	const dispatch = createEventDispatcher();

	import AdvancedParams from './Advanced/AdvancedParams.svelte';
	export let saveSettings: Function;

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
		num_ctx: '',
		num_predict: ''
	};

	const toggleRequestFormat = async () => {
		if (requestFormat === '') {
			requestFormat = 'json';
		} else {
			requestFormat = '';
		}

		saveSettings({ requestFormat: requestFormat !== '' ? requestFormat : undefined });
	};

	onMount(() => {
		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');

		requestFormat = settings.requestFormat ?? '';

		options.seed = settings.seed ?? 0;
		options.temperature = settings.temperature ?? '';
		options.repeat_penalty = settings.repeat_penalty ?? '';
		options.top_k = settings.top_k ?? '';
		options.top_p = settings.top_p ?? '';
		options.num_ctx = settings.num_ctx ?? '';
		options = { ...options, ...settings.options };
		options.stop = (settings?.options?.stop ?? []).join(',');
	});
</script>

<div class="flex flex-col h-full justify-between text-sm">
	<div class=" space-y-3 pr-1.5 overflow-y-scroll max-h-80">
		<div class=" text-sm font-medium">Parameters</div>

		<AdvancedParams bind:options />
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
						stop: options.stop !== '' ? options.stop.split(',').filter((e) => e) : undefined,
						temperature: options.temperature !== '' ? options.temperature : undefined,
						repeat_penalty: options.repeat_penalty !== '' ? options.repeat_penalty : undefined,
						repeat_last_n: options.repeat_last_n !== '' ? options.repeat_last_n : undefined,
						mirostat: options.mirostat !== '' ? options.mirostat : undefined,
						mirostat_eta: options.mirostat_eta !== '' ? options.mirostat_eta : undefined,
						mirostat_tau: options.mirostat_tau !== '' ? options.mirostat_tau : undefined,
						top_k: options.top_k !== '' ? options.top_k : undefined,
						top_p: options.top_p !== '' ? options.top_p : undefined,
						tfs_z: options.tfs_z !== '' ? options.tfs_z : undefined,
						num_ctx: options.num_ctx !== '' ? options.num_ctx : undefined,
						num_predict: options.num_predict !== '' ? options.num_predict : undefined
					}
				});

				dispatch('save');
			}}
		>
			Save
		</button>
	</div>
</div>
