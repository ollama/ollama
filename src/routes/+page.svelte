<script lang="ts">
	import toast from 'svelte-french-toast';
	import Navbar from '$lib/components/layout/Navbar.svelte';

	import { marked } from 'marked';

	import type { PageData } from './$types';
	import { ENDPOINT } from '$lib/contants';

	export let data: PageData;
	$: ({ models } = data);

	let selectedModel = '';
	let prompt = '';
	let context = '';

	let chatHistory = {};

	let textareaElement = '';

	const submitPrompt = async () => {
		console.log('submitPrompt');
		if (selectedModel !== '') {
			console.log(prompt);

			let user_prompt = prompt;
			chatHistory[Object.keys(chatHistory).length] = {
				role: 'user',
				content: user_prompt
			};
			prompt = '';
			textareaElement.style.height = '';

			const res = await fetch(`${ENDPOINT}/api/generate`, {
				method: 'POST',
				headers: {
					'Content-Type': 'text/event-stream'
				},
				body: JSON.stringify({
					model: selectedModel,
					prompt: user_prompt,
					context: context != '' ? context : undefined
				})
			});

			chatHistory[Object.keys(chatHistory).length] = {
				role: 'assistant',
				content: ''
			};

			const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;

				// toast.success(value);
				try {
					let data = JSON.parse(value);
					console.log(data);

					if (data.done == false) {
						if (
							chatHistory[Object.keys(chatHistory).length - 1].content == '' &&
							data.response == '\n'
						) {
							continue;
						} else {
							chatHistory[Object.keys(chatHistory).length - 1].content += data.response;
						}
					} else {
						context = data.context;
						console.log(context);
						chatHistory[Object.keys(chatHistory).length - 1].done = true;
					}
				} catch (error) {
					console.log(error);
				}
				window.scrollTo(0, document.body.scrollHeight);
			}
		} else {
			toast.error('Model not selected');
		}
	};

	const copyToClipboard = (text) => {
		if (!navigator.clipboard) {
			var textArea = document.createElement('textarea');
			textArea.value = text;

			// Avoid scrolling to bottom
			textArea.style.top = '0';
			textArea.style.left = '0';
			textArea.style.position = 'fixed';

			document.body.appendChild(textArea);
			textArea.focus();
			textArea.select();

			try {
				var successful = document.execCommand('copy');
				var msg = successful ? 'successful' : 'unsuccessful';
				console.log('Fallback: Copying text command was ' + msg);
			} catch (err) {
				console.error('Fallback: Oops, unable to copy', err);
			}

			document.body.removeChild(textArea);
			return;
		}
		navigator.clipboard.writeText(text).then(
			function () {
				console.log('Async: Copying to clipboard was successful!');
				toast.success('Copying to clipboard was successful!');
			},
			function (err) {
				console.error('Async: Could not copy text: ', err);
			}
		);
	};
</script>

<div class="app text-gray-100">
	<div class=" bg-gray-800 min-h-screen overflow-auto flex flex-row">
		<Navbar />

		<div class="min-h-screen w-full flex justify-center">
			<div class=" py-2.5 flex flex-col justify-between w-full">
				<div class="max-w-2xl mx-auto w-full px-2.5 mt-14">
					<div class="p-3 rounded-lg bg-gray-900">
						<div>
							<label for="models" class="block mb-2 text-sm font-medium text-gray-200">Model</label>
							<select
								id="models"
								class="outline-none border border-gray-600 bg-gray-700 text-gray-200 text-sm rounded-lg block w-full p-2.5 placeholder-gray-400"
								bind:value={selectedModel}
								disabled={Object.keys(chatHistory).length != 0}
							>
								<option value="" selected>Select a model</option>

								{#each models.models as model}
									<option value={model.name}>{model.name}</option>
								{/each}
							</select>
						</div>
					</div>
				</div>

				<div class=" h-full mb-32 w-full flex flex-col">
					{#if Object.keys(chatHistory).length == 0}
						<div class="m-auto text-4xl text-gray-600 font-bold text-center">Ollama</div>
					{:else}
						{#each Object.keys(chatHistory) as messageIdx}
							<div class=" w-full {chatHistory[messageIdx].role == 'user' ? '' : ' bg-gray-700'}">
								<div class="flex justify-between p-5 py-10 max-w-3xl mx-auto rounded-lg">
									<div class="space-x-7 flex">
										<div class="">
											<img
												src="/{chatHistory[messageIdx].role == 'user' ? 'user' : 'favicon'}.png"
												class=" max-w-[32px] object-cover rounded"
											/>
										</div>

										<div class="whitespace-pre-line">
											{@html marked.parse(chatHistory[messageIdx].content)}
											<!-- {} -->
										</div>
									</div>

									<div>
										{#if chatHistory[messageIdx].role != 'user' && chatHistory[messageIdx].done}
											<button
												class="p-1 rounded hover:bg-gray-700 transition"
												on:click={() => {
													copyToClipboard(chatHistory[messageIdx].content);
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
														d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184"
													/>
												</svg>
											</button>
										{/if}
									</div>
								</div>
							</div>
						{/each}
					{/if}
				</div>
			</div>

			<div class="fixed bottom-0 w-full">
				<!-- <hr class=" mb-3 border-gray-600" /> -->

				<div class=" bg-gradient-to-t from-gray-900 pt-5">
					<div class="max-w-3xl p-2.5 -mb-0.5 mx-auto inset-x-0">
						<form class=" flex shadow-sm relative w-full" on:submit|preventDefault={submitPrompt}>
							<textarea
								class="rounded-xl bg-gray-700 outline-none w-full py-3 px-5 pr-12 resize-none"
								placeholder="Send a message"
								bind:this={textareaElement}
								bind:value={prompt}
								on:keypress={(e) => {
									if (e.keyCode == 13 && !e.shiftKey) {
										e.preventDefault();
									}
									if (prompt !== '' && e.keyCode == 13 && !e.shiftKey) {
										submitPrompt();
									}
								}}
								rows="1"
								on:input={() => {
									textareaElement.style.height = '';
									textareaElement.style.height = Math.min(textareaElement.scrollHeight, 200) + 'px';
								}}
							/>
							<div class=" absolute right-0 bottom-0">
								<div class="pr-3 pb-2">
									<button
										class="{prompt !== ''
											? 'bg-emerald-600 text-gray-100 hover:bg-emerald-700 '
											: 'text-gray-600 disabled'} transition rounded p-2"
										type="submit"
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 16 16"
											fill="none"
											class="w-4 h-4"
											><path
												d="M.5 1.163A1 1 0 0 1 1.97.28l12.868 6.837a1 1 0 0 1 0 1.766L1.969 15.72A1 1 0 0 1 .5 14.836V10.33a1 1 0 0 1 .816-.983L8.5 8 1.316 6.653A1 1 0 0 1 .5 5.67V1.163Z"
												fill="currentColor"
											/></svg
										>
									</button>
								</div>
							</div>
						</form>

						<div class="mt-2.5 text-xs text-gray-500 text-center">
							LLM models may produce inaccurate information about people, places, or facts.
						</div>
					</div>
				</div>
			</div>
		</div>

		<!-- <main class="w-full flex justify-center">
			<div class="max-w-lg w-screen p-5" />
		</main> -->
	</div>
</div>
