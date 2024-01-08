<script lang="ts">
	import toast from 'svelte-french-toast';
	import dayjs from 'dayjs';
	import { onMount } from 'svelte';

	import { getDocs, updateDocByName } from '$lib/apis/documents';
	import Modal from '../common/Modal.svelte';
	import { documents } from '$lib/stores';

	export let show = false;
	export let selectedDoc;

	let doc = {
		name: '',
		title: ''
	};

	const submitHandler = async () => {
		const res = await updateDocByName(localStorage.token, selectedDoc.name, {
			title: doc.title,
			name: doc.name
		}).catch((error) => {
			toast.error(error);
		});

		if (res) {
			show = false;

			documents.set(await getDocs(localStorage.token));
		}
	};

	onMount(() => {
		if (selectedDoc) {
			doc = JSON.parse(JSON.stringify(selectedDoc));
		}
	});
</script>

<Modal size="sm" bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-5 py-4">
			<div class=" text-lg font-medium self-center">Edit Doc</div>
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

		<div class="flex flex-col md:flex-row w-full px-5 py-4 md:space-x-4 dark:text-gray-200">
			<div class=" flex flex-col w-full sm:flex-row sm:justify-center sm:space-x-6">
				<form
					class="flex flex-col w-full"
					on:submit|preventDefault={() => {
						submitHandler();
					}}
				>
					<div class=" flex flex-col space-y-1.5">
						<div class="flex flex-col w-full">
							<div class=" mb-1 text-xs text-gray-500">Name Tag</div>

							<div class="flex flex-1">
								<div
									class="bg-gray-200 dark:bg-gray-600 font-bold px-3 py-1 border border-r-0 dark:border-gray-600 rounded-l-lg flex items-center"
								>
									#
								</div>
								<input
									class="w-full rounded-r-lg py-2.5 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 disabled:text-gray-500 dark:disabled:text-gray-500 outline-none"
									type="text"
									bind:value={doc.name}
									autocomplete="off"
									required
								/>
							</div>

							<!-- <div class="flex-1">
								<input
									class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 disabled:text-gray-500 dark:disabled:text-gray-500 outline-none"
									type="text"
									bind:value={doc.name}
									autocomplete="off"
									required
								/>
							</div> -->
						</div>

						<div class="flex flex-col w-full">
							<div class=" mb-1 text-xs text-gray-500">Title</div>

							<div class="flex-1">
								<input
									class="w-full rounded-lg py-2.5 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
									type="text"
									bind:value={doc.title}
									autocomplete="off"
									required
								/>
							</div>
						</div>
					</div>

					<div class="flex justify-end pt-5 text-sm font-medium">
						<button
							class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
							type="submit"
						>
							Save
						</button>
					</div>
				</form>
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
