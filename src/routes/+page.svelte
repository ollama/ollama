<script lang="ts">
	import toast from 'svelte-french-toast';
	import Navbar from '$lib/components/layout/Navbar.svelte';

	import { v4 as uuidv4 } from 'uuid';
	import { marked } from 'marked';
	import hljs from 'highlight.js';
	import 'highlight.js/styles/dark.min.css';

	import type { PageData } from './$types';
	import { ENDPOINT } from '$lib/contants';
	import { onMount, tick } from 'svelte';

	import { openDB, deleteDB } from 'idb';

	export let data: PageData;
	$: ({ models } = data);
	let textareaElement;
	let db;

	let selectedModel = '';
	let systemPrompt = '';
	let temperature = '';

	let chats = [];
	let chatId = uuidv4();
	let title = ``;
	let prompt = '';
	let messages = [];

	onMount(async () => {
		let settings = localStorage.getItem('settings');
		if (settings) {
			settings = JSON.parse(settings);
			console.log(settings);

			selectedModel = settings.model ?? '';
			systemPrompt = settings.systemPrompt ?? '';
			temperature = settings.temperature ?? '';
		}

		db = await openDB('Chats', 1, {
			upgrade(db) {
				const store = db.createObjectStore('chats', {
					keyPath: 'id',
					autoIncrement: true
				});
				store.createIndex('timestamp', 'timestamp');
			}
		});

		chats = await db.getAllFromIndex('chats', 'timestamp');
		console.log(chats);
		console.log(chatId);
	});

	//////////////////////////
	// Helper functions
	//////////////////////////

	const splitStream = (splitOn) => {
		let buffer = '';
		return new TransformStream({
			transform(chunk, controller) {
				buffer += chunk;
				const parts = buffer.split(splitOn);
				parts.slice(0, -1).forEach((part) => controller.enqueue(part));
				buffer = parts[parts.length - 1];
			},
			flush(controller) {
				if (buffer) controller.enqueue(buffer);
			}
		});
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

	//////////////////////////
	// Web functions
	//////////////////////////

	const saveDefaultModel = () => {
		let settings = localStorage.getItem('settings') ?? '{}';
		if (settings) {
			settings = JSON.parse(settings);
			settings.model = selectedModel;
			localStorage.setItem('settings', JSON.stringify(settings));
		}

		console.log('saved');
		toast.success('Default model updated');
	};

	const createNewChat = () => {
		if (messages.length > 0) {
			messages = [];
			title = '';
			chatId = uuidv4();
		}
	};

	const loadChat = async (id) => {
		const chat = await db.get('chats', id);
		messages = chat.messages;
		title = chat.title;
		chatId = chat.id;
	};

	const deleteChatHistory = async () => {
		const tx = db.transaction('chats', 'readwrite');
		await Promise.all([tx.store.clear(), tx.done]);
		chats = await db.getAllFromIndex('chats', 'timestamp');
	};

	//////////////////////////
	// Ollama functions
	//////////////////////////

	const submitPrompt = async (user_prompt) => {
		console.log('submitPrompt');

		if (selectedModel === '') {
			toast.error('Model not selected');
		} else if (messages.length != 0 && messages.at(-1).done != true) {
			console.log('wait');
		} else {
			if (messages.length == 0) {
				await db.put('chats', {
					id: chatId,
					title: 'New Chat',
					timestamp: Date.now(),
					messages: messages
				});
				chats = await db.getAllFromIndex('chats', 'timestamp');
			}
			messages = [
				...messages,
				{
					role: 'user',
					content: user_prompt
				}
			];
			prompt = '';

			textareaElement.style.height = '';
			setTimeout(() => {
				window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
			}, 50);

			let responseMessage = {
				role: 'assistant',
				content: ''
			};

			messages = [...messages, responseMessage];
			window.scrollTo({ top: document.body.scrollHeight });

			const res = await fetch(`${ENDPOINT}/api/generate`, {
				method: 'POST',
				headers: {
					'Content-Type': 'text/event-stream'
				},
				body: JSON.stringify({
					model: selectedModel,
					prompt: user_prompt,
					context:
						messages.length > 3 && messages.at(-3).context != undefined
							? messages.at(-3).context
							: undefined
				})
			});

			const reader = res.body
				.pipeThrough(new TextDecoderStream())
				.pipeThrough(splitStream('\n'))
				.getReader();

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;

				try {
					let lines = value.split('\n');

					for (const line of lines) {
						if (line !== '') {
							console.log(line);
							let data = JSON.parse(line);
							if (data.done == false) {
								if (responseMessage.content == '' && data.response == '\n') {
									continue;
								} else {
									responseMessage.content += data.response;
									messages = messages;
								}
							} else {
								responseMessage.done = true;
								responseMessage.context = data.context;
								messages = messages;
								hljs.highlightAll();
							}
						}
					}
				} catch (error) {
					console.log(error);
				}
				window.scrollTo({ top: document.body.scrollHeight });
			}

			window.scrollTo({ top: document.body.scrollHeight });

			if (messages.length == 2) {
				await generateTitle(user_prompt);
			}
			await db.put('chats', {
				id: chatId,
				title: title,
				timestamp: Date.now(),
				messages: messages
			});
			chats = await db.getAllFromIndex('chats', 'timestamp');
		}
	};

	const regenerateResponse = async () => {
		console.log('regenerateResponse');

		if (messages.length != 0 && messages.at(-1).done == true) {
			messages.splice(messages.length - 1, 1);
			messages = messages;

			let lastUserMessage = messages.at(-1);

			let responseMessage = {
				role: 'assistant',
				content: ''
			};

			messages = [...messages, responseMessage];
			window.scrollTo({ top: document.body.scrollHeight });

			const res = await fetch(`${ENDPOINT}/api/generate`, {
				method: 'POST',
				headers: {
					'Content-Type': 'text/event-stream'
				},
				body: JSON.stringify({
					model: selectedModel,
					prompt: lastUserMessage.content,
					context:
						messages.length > 3 && messages.at(-3).context != undefined
							? messages.at(-3).context
							: undefined
				})
			});

			const reader = res.body
				.pipeThrough(new TextDecoderStream())
				.pipeThrough(splitStream('\n'))
				.getReader();

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;

				try {
					let lines = value.split('\n');

					for (const line of lines) {
						if (line !== '') {
							console.log(line);
							let data = JSON.parse(line);
							if (data.done == false) {
								if (responseMessage.content == '' && data.response == '\n') {
									continue;
								} else {
									responseMessage.content += data.response;
									messages = messages;
								}
							} else {
								responseMessage.done = true;
								responseMessage.context = data.context;
								messages = messages;
								hljs.highlightAll();
							}
						}
					}
				} catch (error) {
					console.log(error);
				}
				window.scrollTo({ top: document.body.scrollHeight });
			}

			window.scrollTo({ top: document.body.scrollHeight });
			await db.put('chats', {
				id: chatId,
				title: title,
				timestamp: Date.now(),
				messages: messages
			});
			chats = await db.getAllFromIndex('chats', 'timestamp');
		}

		console.log(messages);
	};

	const generateTitle = async (user_prompt) => {
		console.log('generateTitle');

		const res = await fetch(`${ENDPOINT}/api/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream'
			},
			body: JSON.stringify({
				model: selectedModel,
				prompt: `Create an extremely short title for the following question with 3-5 words without using the word 'title.': ${user_prompt}`,
				stream: false
			})
		})
			.then(async (res) => {
				if (!res.ok) throw await res.json();
				return res.json();
			})
			.catch((error) => {
				console.log(error);
				return null;
			});

		if (res) {
			console.log(res);
			title = res.response;
		}
	};
</script>

<div class="app text-gray-100">
	<div class=" bg-gray-800 min-h-screen overflow-auto flex flex-row">
		<Navbar {chats} {title} {loadChat} {createNewChat} {deleteChatHistory} />

		<div class="min-h-screen w-full flex justify-center">
			<div class=" py-2.5 flex flex-col justify-between w-full">
				<div class="max-w-2xl mx-auto w-full px-2.5 mt-14">
					<div class="p-3 rounded-lg bg-gray-900">
						<div>
							<label for="models" class="block mb-2 text-sm font-medium text-gray-200">Model</label>

							<div>
								<select
									id="models"
									class="outline-none border border-gray-600 bg-gray-700 text-gray-200 text-sm rounded-lg block w-full p-2.5 placeholder-gray-400"
									bind:value={selectedModel}
									disabled={messages.length != 0}
								>
									<option value="" selected>Select a model</option>

									{#each models.models as model}
										<option value={model.name}>{model.name}</option>
									{/each}
								</select>
								<div class="text-right mt-1.5 text-xs text-gray-500">
									<button on:click={saveDefaultModel}> Set as default</button>
								</div>
							</div>
						</div>
					</div>
				</div>

				<div class=" h-full mb-48 w-full flex flex-col">
					{#if messages.length == 0}
						<div class="m-auto text-center max-w-md pb-16">
							<div class="flex justify-center mt-8">
								<img src="/ollama.png" class="w-16 invert-[80%]" />
							</div>
							<div class="mt-6 text-3xl text-gray-500 font-semibold">
								Get up and running with large language models, locally.
							</div>

							<div class=" my-4 text-gray-600">
								Run Llama 2, Code Llama, and other models. <br /> Customize and create your own.
							</div>
						</div>
					{:else}
						{#each messages as message, messageIdx}
							<div class=" w-full {message.role == 'user' ? '' : ' bg-gray-700'}">
								<div class="flex justify-between p-5 py-10 max-w-3xl mx-auto rounded-lg">
									<div class="space-x-7 flex w-full">
										<div class="">
											<img
												src="/{message.role == 'user' ? 'user' : 'favicon'}.png"
												class=" max-w-[32px] object-cover rounded"
											/>
										</div>

										{#if message.role != 'user' && message.content == ''}
											<div class="w-full pr-28">
												<div class="animate-pulse flex w-full">
													<div class="space-y-2 w-full">
														<div class="h-2 bg-gray-600 rounded mr-14" />

														<div class="grid grid-cols-3 gap-4">
															<div class="h-2 bg-gray-600 rounded col-span-2" />
															<div class="h-2 bg-gray-600 rounded col-span-1" />
														</div>
														<div class="grid grid-cols-4 gap-4">
															<div class="h-2 bg-gray-600 rounded col-span-1" />
															<div class="h-2 bg-gray-600 rounded col-span-2" />
															<div class="h-2 bg-gray-600 rounded col-span-1 mr-4" />
														</div>

														<div class="h-2 bg-gray-600 rounded" />
													</div>
												</div>
											</div>
										{:else}
											<div class="whitespace-pre-line">
												{@html marked.parse(message.content)}
											</div>
										{/if}
										<!-- {} -->
									</div>

									<div>
										{#if message.role != 'user' && message.done}
											<button
												class="p-1 rounded hover:bg-gray-700 transition"
												on:click={() => {
													copyToClipboard(message.content);
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
						{#if messages.length == 0}
							<div class=" grid sm:grid-cols-2 gap-2.5 mb-4 md:p-2 text-left">
								<button
									class=" flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
									on:click={() => {
										submitPrompt(`Tell me a random fun fact about the Roman Empire`);
									}}
								>
									<div class="flex flex-col text-left">
										<div class="text-sm font-medium text-gray-300">Tell me a fun fact</div>
										<div class="text-sm text-gray-500">about the Roman Empire</div>
									</div>

									<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
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
									</div>
								</button>

								<button
									class="flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
									on:click={() => {
										submitPrompt(
											`Show me a code snippet of a website's sticky header in CSS and JavaScript.`
										);
									}}
								>
									<div class="flex flex-col text-left">
										<div class="text-sm font-medium text-gray-300">Show me a code snippet</div>
										<div class="text-sm text-gray-500">of a website's sticky header</div>
									</div>
									<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
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
									</div>
								</button>

								<button
									class=" hidden sm:flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
									on:click={() => {
										submitPrompt(
											`Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option.`
										);
									}}
								>
									<div class="flex flex-col text-left">
										<div class="text-sm font-medium text-gray-300">Help me study</div>
										<div class="text-sm text-gray-500">vocabulary for a college entrance exam</div>
									</div>
									<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
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
									</div>
								</button>

								<button
									class="  hidden sm:flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
									on:click={() => {
										submitPrompt(
											`What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.`
										);
									}}
								>
									<div class="flex flex-col text-left">
										<div class="text-sm font-medium text-gray-300">Give me ideas</div>
										<div class="text-sm text-gray-500">for what to do with my kids' art</div>
									</div>
									<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
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
									</div>
								</button>
							</div>
						{/if}

						{#if messages.length != 0 && messages.at(-1).role == 'assistant' && messages.at(-1).done == true}
							<div class=" flex justify-end mb-2.5">
								<button
									class=" flex px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg"
									on:click={regenerateResponse}
								>
									<div class=" self-center mr-1">
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 20 20"
											fill="currentColor"
											class="w-4 h-4"
										>
											<path
												fill-rule="evenodd"
												d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z"
												clip-rule="evenodd"
											/>
										</svg>
									</div>
									<div class=" self-center text-sm">Regenerate</div>
								</button>
							</div>
						{/if}
						<form
							class=" flex shadow-sm relative w-full"
							on:submit|preventDefault={() => {
								submitPrompt(prompt);
							}}
						>
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
										submitPrompt(prompt);
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
									{#if messages.length == 0 || messages.at(-1).done == true}
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
									{:else}
										<div class="loading mb-1.5 mr-1 font-semibold text-lg">...</div>
									{/if}
								</div>
							</div>
						</form>

						<div class="mt-2.5 text-xs text-gray-500 text-center">
							LLMs may produce inaccurate information about people, places, or facts.
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

<style>
	.loading {
		display: inline-block;
		clip-path: inset(0 1ch 0 0);
		animation: l 1s steps(3) infinite;
		letter-spacing: -0.5px;
	}

	@keyframes l {
		to {
			clip-path: inset(0 -1ch 0 0);
		}
	}
</style>
