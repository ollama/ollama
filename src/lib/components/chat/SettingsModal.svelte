<script lang="ts">
	import { onMount } from 'svelte';
	import Modal from '../common/Modal.svelte';
	export let show = false;
	export let saveSettings: Function;
	let system = '';
	let temperature = 0.8;

	$: if (show) {
		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');
		system = settings.system ?? '';
		temperature = settings.temperature ?? 0.8;
	}
</script>

<Modal bind:show>
	<div class="rounded-lg bg-gray-900">
		<div class=" flex justify-between text-gray-300 px-5 py-4">
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
		<hr class=" border-gray-800" />

		<div class="p-5 flex flex-col space-y-3">
			<div>
				<div class=" mb-2.5 text-sm font-medium">System Prompt</div>
				<textarea
					bind:value={system}
					class="w-full rounded p-4 text-sm text-gray-300 bg-gray-800 outline-none"
					rows="4"
				/>
			</div>

			<div>
				<label for="steps-range" class=" mb-2 text-sm font-medium flex justify-between">
					<div>Temperature</div>
					<div>
						{temperature}
					</div></label
				>
				<input
					id="steps-range"
					type="range"
					min="0"
					max="1"
					bind:value={temperature}
					step="0.05"
					class="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-700"
				/>
			</div>
			<div class="flex justify-end pt-3 text-sm font-medium">
				<button
					class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 transition rounded"
					on:click={() => {
						saveSettings(system != '' ? system : null, temperature != 0.8 ? temperature : null);
						show = false;
					}}
				>
					Save
				</button>
			</div>
		</div>
	</div>
</Modal>
