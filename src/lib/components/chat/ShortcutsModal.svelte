<script lang="ts">
	import Modal from '../common/Modal.svelte';

	export let show = false;
</script>

<Modal bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-5 py-4">
			<div class=" text-lg font-medium self-center">Keyboard shortcuts</div>
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
		<hr class=" dark:border-gray-800" />

		<div class="flex flex-col md:flex-row w-full p-5 md:space-x-4 dark:text-gray-200">
			<div class=" flex flex-col w-full sm:flex-row sm:justify-center sm:space-x-6">
				<div class="flex flex-col space-y-3 w-full self-start">
					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Open new chat</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Shift
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								O
							</div>
						</div>
					</div>

					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Focus chat input</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Shift
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Esc
							</div>
						</div>
					</div>

					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Copy last code block</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Shift
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								;
							</div>
						</div>
					</div>

					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Copy last response</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Shift
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								C
							</div>
						</div>
					</div>
				</div>

				<div class="flex flex-col space-y-3 w-full self-start">
					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Toggle settings</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								.
							</div>
						</div>
					</div>

					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Toggle sidebar</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Shift
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								S
							</div>
						</div>
					</div>

					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Delete chat</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Shift
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								⌫
							</div>
						</div>
					</div>

					<div class="w-full flex justify-between items-center">
						<div class=" text-sm">Show shortcuts</div>

						<div class="flex space-x-1 text-xs">
							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								Ctrl/⌘
							</div>

							<div
								class=" h-fit py-1 px-2 flex items-center justify-center rounded border border-black/10 capitalize text-gray-600 dark:border-white/10 dark:text-gray-300"
							>
								/
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</Modal>

<style>
	input::-webkit-outer-spin-button,
	input::-webkit-inner-spin-button {
		/* display: none; <- Crashes Chrome on hover */
		-webkit-appearance: none;
		margin: 0; /* <-- Apparently some margin are still there even though it's hidden */
	}

	.tabs::-webkit-scrollbar {
		display: none; /* for Chrome, Safari and Opera */
	}

	.tabs {
		-ms-overflow-style: none; /* IE and Edge */
		scrollbar-width: none; /* Firefox */
	}

	input[type='number'] {
		-moz-appearance: textfield; /* Firefox */
	}
</style>
