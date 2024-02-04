<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	import { documents } from '$lib/stores';
	import { removeFirstHashWord, isValidHttpUrl } from '$lib/utils';
	import { tick } from 'svelte';
	import toast from 'svelte-french-toast';

	export let prompt = '';

	const dispatch = createEventDispatcher();
	let selectedIdx = 0;

	let filteredItems = [];
	let filteredDocs = [];

	let collections = [];

	$: collections = [
		...($documents.length > 0
			? [
					{
						name: 'All Documents',
						type: 'collection',
						title: 'All Documents',
						collection_names: $documents.map((doc) => doc.collection_name)
					}
			  ]
			: []),
		...$documents
			.reduce((a, e, i, arr) => {
				return [...new Set([...a, ...(e?.content?.tags ?? []).map((tag) => tag.name)])];
			}, [])
			.map((tag) => ({
				name: tag,
				type: 'collection',
				collection_names: $documents
					.filter((doc) => (doc?.content?.tags ?? []).map((tag) => tag.name).includes(tag))
					.map((doc) => doc.collection_name)
			}))
	];

	$: filteredCollections = collections
		.filter((collection) => collection.name.includes(prompt.split(' ')?.at(0)?.substring(1) ?? ''))
		.sort((a, b) => a.name.localeCompare(b.name));

	$: filteredDocs = $documents
		.filter((doc) => doc.name.includes(prompt.split(' ')?.at(0)?.substring(1) ?? ''))
		.sort((a, b) => a.title.localeCompare(b.title));

	$: filteredItems = [...filteredCollections, ...filteredDocs];

	$: if (prompt) {
		selectedIdx = 0;

		console.log(filteredCollections);
	}

	export const selectUp = () => {
		selectedIdx = Math.max(0, selectedIdx - 1);
	};

	export const selectDown = () => {
		selectedIdx = Math.min(selectedIdx + 1, filteredItems.length - 1);
	};

	const confirmSelect = async (doc) => {
		dispatch('select', doc);

		prompt = removeFirstHashWord(prompt);
		const chatInputElement = document.getElementById('chat-textarea');

		await tick();
		chatInputElement?.focus();
		await tick();
	};

	const confirmSelectWeb = async (url) => {
		dispatch('url', url);

		prompt = removeFirstHashWord(prompt);
		const chatInputElement = document.getElementById('chat-textarea');

		await tick();
		chatInputElement?.focus();
		await tick();
	};
</script>

{#if filteredItems.length > 0 || prompt.split(' ')?.at(0)?.substring(1).startsWith('http')}
	<div class="md:px-2 mb-3 text-left w-full">
		<div class="flex w-full rounded-lg border border-gray-100 dark:border-gray-700">
			<div class=" bg-gray-100 dark:bg-gray-700 w-10 rounded-l-lg text-center">
				<div class=" text-lg font-semibold mt-2">#</div>
			</div>

			<div class="max-h-60 flex flex-col w-full rounded-r-lg">
				<div class=" overflow-y-auto bg-white p-2 rounded-tr-lg space-y-0.5">
					{#each filteredItems as doc, docIdx}
						<button
							class=" px-3 py-1.5 rounded-lg w-full text-left {docIdx === selectedIdx
								? ' bg-gray-100 selected-command-option-button'
								: ''}"
							type="button"
							on:click={() => {
								console.log(doc);

								confirmSelect(doc);
							}}
							on:mousemove={() => {
								selectedIdx = docIdx;
							}}
							on:focus={() => {}}
						>
							{#if doc.type === 'collection'}
								<div class=" font-medium text-black line-clamp-1">
									{doc?.title ?? `#${doc.name}`}
								</div>

								<div class=" text-xs text-gray-600 line-clamp-1">Collection</div>
							{:else}
								<div class=" font-medium text-black line-clamp-1">
									#{doc.name} ({doc.filename})
								</div>

								<div class=" text-xs text-gray-600 line-clamp-1">
									{doc.title}
								</div>
							{/if}
						</button>
					{/each}

					{#if prompt.split(' ')?.at(0)?.substring(1).startsWith('http')}
						<button
							class="px-3 py-1.5 rounded-lg w-full text-left bg-gray-100 selected-command-option-button"
							type="button"
							on:click={() => {
								const url = prompt.split(' ')?.at(0)?.substring(1);
								if (isValidHttpUrl(url)) {
									confirmSelectWeb(url);
								} else {
									toast.error(
										'Oops! Looks like the URL is invalid. Please double-check and try again.'
									);
								}
							}}
						>
							<div class=" font-medium text-black line-clamp-1">
								{prompt.split(' ')?.at(0)?.substring(1)}
							</div>

							<div class=" text-xs text-gray-600 line-clamp-1">Web</div>
						</button>
					{/if}
				</div>
			</div>
		</div>
	</div>
{/if}
