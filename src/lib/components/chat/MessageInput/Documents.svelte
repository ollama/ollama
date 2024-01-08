<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	import { documents } from '$lib/stores';
	import { removeFirstHashWord } from '$lib/utils';
	import { tick } from 'svelte';

	export let prompt = '';

	const dispatch = createEventDispatcher();
	let selectedIdx = 0;
	let filteredDocs = [];

	$: filteredDocs = $documents
		.filter((p) => p.name.includes(prompt.split(' ')?.at(0)?.substring(1) ?? ''))
		.sort((a, b) => a.title.localeCompare(b.title));

	$: if (prompt) {
		selectedIdx = 0;
	}

	export const selectUp = () => {
		selectedIdx = Math.max(0, selectedIdx - 1);
	};

	export const selectDown = () => {
		selectedIdx = Math.min(selectedIdx + 1, filteredDocs.length - 1);
	};

	const confirmSelect = async (doc) => {
		dispatch('select', doc);

		prompt = removeFirstHashWord(prompt);
		const chatInputElement = document.getElementById('chat-textarea');

		await tick();
		chatInputElement?.focus();
		await tick();
	};
</script>

{#if filteredDocs.length > 0}
	<div class="md:px-2 mb-3 text-left w-full">
		<div class="flex w-full rounded-lg border border-gray-100 dark:border-gray-700">
			<div class=" bg-gray-100 dark:bg-gray-700 w-10 rounded-l-lg text-center">
				<div class=" text-lg font-semibold mt-2">#</div>
			</div>

			<div class="max-h-60 flex flex-col w-full rounded-r-lg">
				<div class=" overflow-y-auto bg-white p-2 rounded-tr-lg space-y-0.5">
					{#each filteredDocs as doc, docIdx}
						<button
							class=" px-3 py-1.5 rounded-lg w-full text-left {docIdx === selectedIdx
								? ' bg-gray-100 selected-command-option-button'
								: ''}"
							type="button"
							on:click={() => {
								confirmSelect(doc);
							}}
							on:mousemove={() => {
								selectedIdx = docIdx;
							}}
							on:focus={() => {}}
						>
							<div class=" font-medium text-black line-clamp-1">
								#{doc.name} ({doc.filename})
							</div>

							<div class=" text-xs text-gray-600 line-clamp-1">
								{doc.title}
							</div>
						</button>
					{/each}
				</div>
			</div>
		</div>
	</div>
{/if}
