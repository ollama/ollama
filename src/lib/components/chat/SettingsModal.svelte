<script lang="ts">
	import { onMount } from 'svelte';
	import Modal from '../common/Modal.svelte';
	export let show = false;
	export let saveSettings: Function;
	let system = '';
	let temperature = 0.8;

	let selectedMenu = 'general';

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

		<div class="flex flex-col md:flex-row w-full p-4 md:space-x-4">
			<div
				class="flex flex-row space-x-1 md:space-x-0 md:space-y-1 md:flex-col flex-1 md:flex-none md:w-36 text-gray-200 text-xs text-left mb-3 md:mb-0"
			>
				<button
					class="px-2 py-2 rounded flex-1 md:flex-none flex text-right transition {selectedMenu ===
					'general'
						? 'bg-gray-700'
						: 'hover:bg-gray-800'}"
					on:click={() => {
						selectedMenu = 'general';
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
					class="px-2 py-2 rounded flex-1 md:flex-none flex text-right transition {selectedMenu ===
					'models'
						? 'bg-gray-700'
						: 'hover:bg-gray-800'}"
					on:click={() => {
						selectedMenu = 'models';
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
			</div>
			<div class="flex-1 md:min-h-[280px]">
				{#if selectedMenu === 'general'}
					<div class="flex flex-col space-y-3">
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
									saveSettings(
										system != '' ? system : null,
										temperature != 0.8 ? temperature : null
									);
									show = false;
								}}
							>
								Save
							</button>
						</div>
					</div>
				{:else if selectedMenu === 'models'}
					<div class="text-sm">
						<div>
							<div>Pull a model</div>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
</Modal>
