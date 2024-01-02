<script lang="ts">
	import { findWordIndices } from '$lib/utils';
	import { tick } from 'svelte';

	export let prompt = '';

	let selectedCommandIdx = 0;

	let promptCommands = [
		{
			command: '/article',
			title: 'Article Generator',
			content: `Write an article about [topic]

include relevant statistics (add the links of the sources you use) and consider diverse perspectives. Write it in a [X_tone] and mention the source links in the end.`
		},
		{
			command: '/backlink',

			title: 'Backlink Outreach Email',
			content: `Write a link-exchange outreach email on behalf of [your name] from [your_company] to ask for a backlink from their [website_url] to [your website url].`
		},
		{
			command: '/faq',

			title: 'FAQ Generator',
			content: `Create a list of [10] frequently asked questions about [keyword] and provide answers for each one of them considering the SERP and rich result guidelines.`
		},
		{
			command: '/headline',

			title: 'Headline Generator',
			content: `Generate 10 attention-grabbing headlines for an article about [your topic]`
		},
		{
			command: '/product',

			title: 'Product Description',
			content: `Craft an irresistible product description that highlights the benefits of [your product]`
		},
		{
			command: '/seo',

			title: 'SEO Content Brief',
			content: `Create a SEO content brief for [keyword].`
		}
	];

	let filteredPromptCommands = [];

	$: filteredPromptCommands = promptCommands.filter((p) => p.command.includes(prompt));

	$: if (prompt) {
		selectedCommandIdx = 0;
	}

	export const selectUp = () => {
		selectedCommandIdx = Math.max(0, selectedCommandIdx - 1);
	};

	export const selectDown = () => {
		selectedCommandIdx = Math.min(selectedCommandIdx + 1, filteredPromptCommands.length - 1);
	};

	const confirmCommand = async (command) => {
		prompt = command.content;

		const chatInputElement = document.getElementById('chat-textarea');

		await tick();

		chatInputElement.style.height = '';
		chatInputElement.style.height = Math.min(chatInputElement.scrollHeight, 200) + 'px';

		chatInputElement?.focus();

		await tick();

		const words = findWordIndices(prompt);

		if (words.length > 0) {
			const word = words.at(0);
			chatInputElement.setSelectionRange(word?.startIndex, word.endIndex + 1);
		}
	};
</script>

{#if filteredPromptCommands.length > 0}
	<div class="md:px-2 mb-3 text-left w-full">
		<div class="flex w-full rounded-lg border border-gray-100 dark:border-gray-700">
			<div class=" bg-gray-100 dark:bg-gray-700 w-10 rounded-l-lg text-center">
				<div class=" text-lg font-medium mt-2">/</div>
			</div>
			<div class=" max-h-60 overflow-y-auto bg-white w-full p-2 rounded-r-lg space-y-0.5">
				{#each filteredPromptCommands as command, commandIdx}
					<button
						class=" px-3 py-1.5 rounded-lg w-full text-left {commandIdx === selectedCommandIdx
							? ' bg-gray-100 selected-command-option-button'
							: ''}"
						type="button"
						on:click={() => {
							confirmCommand(command);
						}}
						on:mousemove={() => {
							selectedCommandIdx = commandIdx;
						}}
						on:focus={() => {}}
					>
						<div class=" font-medium text-black">
							{command.command}
						</div>

						<div class=" text-xs text-gray-600">
							{command.title}
						</div>
					</button>
				{/each}
			</div>
		</div>
	</div>
{/if}
