<script lang="ts">
	import { generatePrompt } from '$lib/apis/ollama';
	import { models } from '$lib/stores';
	import { splitStream } from '$lib/utils';
	import { tick } from 'svelte';
	import toast from 'svelte-french-toast';

	export let prompt = '';
	export let user = null;

	export let chatInputPlaceholder = '';
	export let messages = [];

	let selectedIdx = 0;
	let filteredModels = [];

	$: filteredModels = $models
		.filter(
			(p) =>
				p.name !== 'hr' &&
				!p.external &&
				p.name.includes(prompt.split(' ')?.at(0)?.substring(1) ?? '')
		)
		.sort((a, b) => a.name.localeCompare(b.name));

	$: if (prompt) {
		selectedIdx = 0;
	}

	export const selectUp = () => {
		selectedIdx = Math.max(0, selectedIdx - 1);
	};

	export const selectDown = () => {
		selectedIdx = Math.min(selectedIdx + 1, filteredModels.length - 1);
	};

	const confirmSelect = async (model) => {
		// dispatch('select', model);
		prompt = '';
		user = JSON.parse(JSON.stringify(model.name));
		await tick();

		chatInputPlaceholder = `'${model.name}' is thinking...`;

		const chatInputElement = document.getElementById('chat-textarea');

		await tick();
		chatInputElement?.focus();
		await tick();

		const convoText = messages.reduce((a, message, i, arr) => {
			return `${a}### ${message.role.toUpperCase()}\n${message.content}\n\n`;
		}, '');

		const res = await generatePrompt(localStorage.token, model.name, convoText);

		if (res && res.ok) {
			const reader = res.body
				.pipeThrough(new TextDecoderStream())
				.pipeThrough(splitStream('\n'))
				.getReader();

			while (true) {
				const { value, done } = await reader.read();
				if (done) {
					break;
				}

				try {
					let lines = value.split('\n');

					for (const line of lines) {
						if (line !== '') {
							console.log(line);
							let data = JSON.parse(line);

							if ('detail' in data) {
								throw data;
							}

							if (data.done == false) {
								if (prompt == '' && data.response == '\n') {
									continue;
								} else {
									prompt += data.response;
									console.log(data.response);
									chatInputElement.scrollTop = chatInputElement.scrollHeight;
									await tick();
								}
							}
						}
					}
				} catch (error) {
					console.log(error);
					if ('detail' in error) {
						toast.error(error.detail);
					}
					break;
				}
			}
		} else {
			if (res !== null) {
				const error = await res.json();
				console.log(error);
				if ('detail' in error) {
					toast.error(error.detail);
				} else {
					toast.error(error.error);
				}
			} else {
				toast.error(`Uh-oh! There was an issue connecting to Ollama.`);
			}
		}

		chatInputPlaceholder = '';

		console.log(user);
	};
</script>

{#if filteredModels.length > 0}
	<div class="md:px-2 mb-3 text-left w-full">
		<div class="flex w-full rounded-lg border border-gray-100 dark:border-gray-700">
			<div class=" bg-gray-100 dark:bg-gray-700 w-10 rounded-l-lg text-center">
				<div class=" text-lg font-semibold mt-2">@</div>
			</div>

			<div class="max-h-60 flex flex-col w-full rounded-r-lg">
				<div class=" overflow-y-auto bg-white p-2 rounded-tr-lg space-y-0.5">
					{#each filteredModels as model, modelIdx}
						<button
							class=" px-3 py-1.5 rounded-lg w-full text-left {modelIdx === selectedIdx
								? ' bg-gray-100 selected-command-option-button'
								: ''}"
							type="button"
							on:click={() => {
								confirmSelect(model);
							}}
							on:mousemove={() => {
								selectedIdx = modelIdx;
							}}
							on:focus={() => {}}
						>
							<div class=" font-medium text-black line-clamp-1">
								{model.name}
							</div>

							<!-- <div class=" text-xs text-gray-600 line-clamp-1">
								{doc.title}
							</div> -->
						</button>
					{/each}
				</div>
			</div>
		</div>
	</div>
{/if}
