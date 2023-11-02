<script lang="ts">
	import { openDB, deleteDB } from 'idb';
	import { v4 as uuidv4 } from 'uuid';
	import { marked } from 'marked';
	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;
	import hljs from 'highlight.js';
	import 'highlight.js/styles/github-dark.min.css';
	import katex from 'katex';
	import auto_render from 'katex/dist/contrib/auto-render.mjs';
	import toast from 'svelte-french-toast';

	import { API_BASE_URL as BUILD_TIME_API_BASE_URL } from '$lib/constants';
	import { onMount, tick } from 'svelte';

	import Navbar from '$lib/components/layout/Navbar.svelte';
	import SettingsModal from '$lib/components/chat/SettingsModal.svelte';
	import Suggestions from '$lib/components/chat/Suggestions.svelte';

	let API_BASE_URL = BUILD_TIME_API_BASE_URL;
	let suggestions = ''; // $page.url.searchParams.get('suggestions');

	let models = [];
	let textareaElement;

	let showSettings = false;
	let db;

	let selectedModel = '';
	let system = null;
	let temperature = null;

	let chats = [];
	let chatId = uuidv4();
	let title = '';
	let prompt = '';
	let messages = [];

	let stopResponseFlag = false;
	let autoScroll = true;

	onMount(async () => {
		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');

		API_BASE_URL = settings.API_BASE_URL ?? BUILD_TIME_API_BASE_URL;
		console.log(API_BASE_URL);
		system = settings.system ?? null;
		temperature = settings.temperature ?? null;

		await getModelTags();

		selectedModel =
			settings.model && models.map((model) => model.name).includes(settings.model)
				? settings.model
				: '';

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

	const createCopyCodeBlockButton = () => {
		// use a class selector if available
		let blocks = document.querySelectorAll('pre');
		console.log(blocks);

		blocks.forEach((block) => {
			// only add button if browser supports Clipboard API

			if (navigator.clipboard && block.childNodes.length < 2) {
				let button = document.createElement('button');

				button.innerText = 'Copy Code';
				block.appendChild(button);

				button.addEventListener('click', async () => {
					await copyCode(block, button);
				});
			}
		});

		async function copyCode(block, button) {
			let code = block.querySelector('code');
			let text = code.innerText;

			await navigator.clipboard.writeText(text);

			// visual feedback that task is completed
			button.innerText = 'Code Copied';

			setTimeout(() => {
				button.innerText = 'Copy Code';
			}, 700);
		}
	};

	const renderLatex = () => {
		let chatMessageElements = document.getElementsByClassName('chat-assistant');
		// let lastChatMessageElement = chatMessageElements[chatMessageElements.length - 1];

		for (const element of chatMessageElements) {
			auto_render(element, {
				// customised options
				// • auto-render specific keys, e.g.:
				delimiters: [
					{ left: '$$', right: '$$', display: true },
					{ left: '$', right: '$', display: false },
					{ left: '\\(', right: '\\)', display: false },
					{ left: '\\[', right: '\\]', display: true }
				],
				// • rendering keys, e.g.:
				throwOnError: false,
				output: 'mathml'
			});
		}
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

	const saveSettings = async (_api_base_url, _system, _temperature) => {
		API_BASE_URL = _api_base_url;
		system = _system;
		temperature = _temperature;

		let settings = localStorage.getItem('settings') ?? '{}';
		if (settings) {
			settings = JSON.parse(settings);

			settings.API_BASE_URL = API_BASE_URL;
			settings.system = system;
			settings.temperature = temperature;
			localStorage.setItem('settings', JSON.stringify(settings));
		}

		console.log(settings);
		await getModelTags();
	};

	const createNewChat = () => {
		if (messages.length > 0) {
			chatId = uuidv4();

			messages = [];
			title = '';
			console.log(localStorage.settings.model);

			let settings = localStorage.getItem('settings');
			if (settings) {
				settings = JSON.parse(settings);
				console.log(settings);

				selectedModel =
					settings.model && models.map((model) => model.name).includes(settings.model)
						? settings.model
						: '';
				system = settings.system ?? system;
				temperature = settings.temperature ?? temperature;
			}
		}
	};

	const loadChat = async (id) => {
		const chat = await db.get('chats', id);
		if (chatId !== chat.id) {
			if (chat.messages.length > 0) {
				chat.messages.at(-1).done = true;
			}
			messages = chat.messages;
			title = chat.title;
			chatId = chat.id;
			selectedModel = chat.model ?? selectedModel;
			system = chat.system ?? system;
			temperature = chat.temperature ?? temperature;

			await tick();
			hljs.highlightAll();
			createCopyCodeBlockButton();
			renderLatex();
		}
	};

	const editChatTitle = async (id, _title) => {
		const chat = await db.get('chats', id);
		console.log(chat);

		await db.put('chats', {
			...chat,
			title: _title
		});

		title = _title;
		chats = await db.getAllFromIndex('chats', 'timestamp');
	};

	const deleteChat = async (id) => {
		createNewChat();

		const chat = await db.delete('chats', id);
		console.log(chat);
		chats = await db.getAllFromIndex('chats', 'timestamp');
	};

	const deleteChatHistory = async () => {
		const tx = db.transaction('chats', 'readwrite');
		await Promise.all([tx.store.clear(), tx.done]);
		chats = await db.getAllFromIndex('chats', 'timestamp');
	};

	const importChatHistory = async (results) => {
		for (const chat of results) {
			console.log(chat);

			await db.put('chats', {
				id: chat.id,
				model: chat.model,
				system: chat.system,
				options: chat.options,
				title: chat.title,
				timestamp: chat.timestamp,
				messages: chat.messages
			});
		}
		chats = await db.getAllFromIndex('chats', 'timestamp');

		console.log(chats);
	};

	const exportChatHistory = async () => {
		chats = await db.getAllFromIndex('chats', 'timestamp');
		let blob = new Blob([JSON.stringify(chats)], { type: 'application/json' });
		saveAs(blob, `chat-export-${Date.now()}.json`);
	};

	const openSettings = async () => {
		showSettings = true;
	};

	const editMessage = async (messageIdx) => {
		messages = messages.map((message, idx) => {
			if (messageIdx === idx) {
				message.edit = true;
				message.editedContent = message.content;
			}
			return message;
		});
	};

	const confirmEditMessage = async (messageIdx) => {
		let userPrompt = messages.at(messageIdx).editedContent;

		messages.splice(messageIdx, messages.length - messageIdx);
		messages = messages;

		await submitPrompt(userPrompt);
	};

	const cancelEditMessage = (messageIdx) => {
		messages = messages.map((message, idx) => {
			if (messageIdx === idx) {
				message.edit = undefined;
				message.editedContent = undefined;
			}
			return message;
		});

		console.log(messages);
	};

	const rateMessage = async (messageIdx, rating) => {
		messages = messages.map((message, idx) => {
			if (messageIdx === idx) {
				message.rating = rating;
			}
			return message;
		});

		await db.put('chats', {
			id: chatId,
			title: title === '' ? 'New Chat' : title,
			model: selectedModel,
			system: system,
			options: {
				temperature: temperature
			},
			timestamp: Date.now(),
			messages: messages
		});

		console.log(messages);
	};

	//////////////////////////
	// Ollama functions
	//////////////////////////

	const getModelTags = async (url = null) => {
		const res = await fetch(`${url === null ? API_BASE_URL : url}/tags`, {
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json'
			}
		})
			.then(async (res) => {
				if (!res.ok) throw await res.json();
				return res.json();
			})
			.catch((error) => {
				console.log(error);
				toast.error('Server connection failed');
				return null;
			});

		console.log(res);
		models = res?.models ?? [];
		return res;
	};

	const sendPrompt = async (userPrompt) => {
		let responseMessage = {
			role: 'assistant',
			content: ''
		};

		messages = [...messages, responseMessage];
		window.scrollTo({ top: document.body.scrollHeight });

		const res = await fetch(`${API_BASE_URL}/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream'
			},
			body: JSON.stringify({
				model: selectedModel,
				prompt: userPrompt,
				system: system ?? undefined,
				options:
					temperature != null
						? {
								temperature: temperature
						  }
						: undefined,
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
			if (done || stopResponseFlag) {
				if (stopResponseFlag) {
					responseMessage.done = true;
					messages = messages;
					hljs.highlightAll();
					createCopyCodeBlockButton();
					renderLatex();
				}

				break;
			}

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
							createCopyCodeBlockButton();
							renderLatex();
						}
					}
				}
			} catch (error) {
				console.log(error);
			}

			if (autoScroll) {
				window.scrollTo({ top: document.body.scrollHeight });
			}

			await db.put('chats', {
				id: chatId,
				title: title === '' ? 'New Chat' : title,
				model: selectedModel,
				system: system,
				options: {
					temperature: temperature
				},
				timestamp: Date.now(),
				messages: messages
			});
		}

		stopResponseFlag = false;
		await tick();
		if (autoScroll) {
			window.scrollTo({ top: document.body.scrollHeight });
		}
	};

	const submitPrompt = async (userPrompt) => {
		console.log('submitPrompt');

		if (selectedModel === '') {
			toast.error('Model not selected');
		} else if (messages.length != 0 && messages.at(-1).done != true) {
			console.log('wait');
		} else {
			if (messages.length == 0) {
				await db.put('chats', {
					id: chatId,
					model: selectedModel,
					system: system,
					options: {
						temperature: temperature
					},
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
					content: userPrompt
				}
			];

			prompt = '';
			textareaElement.style.height = '';

			setTimeout(() => {
				window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
			}, 50);

			await sendPrompt(userPrompt);

			if (messages.length == 2) {
				await generateTitle(chatId, userPrompt);
			}
			chats = await db.getAllFromIndex('chats', 'timestamp');
		}
	};

	const regenerateResponse = async () => {
		console.log('regenerateResponse');
		if (messages.length != 0 && messages.at(-1).done == true) {
			messages.splice(messages.length - 1, 1);
			messages = messages;

			let userMessage = messages.at(-1);
			let userPrompt = userMessage.content;
			await sendPrompt(userPrompt);
			chats = await db.getAllFromIndex('chats', 'timestamp');
		}
	};

	const stopResponse = () => {
		stopResponseFlag = true;
		console.log('stopResponse');
	};

	const generateTitle = async (_chatId, userPrompt) => {
		console.log('generateTitle');

		const res = await fetch(`${API_BASE_URL}/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream'
			},
			body: JSON.stringify({
				model: selectedModel,
				prompt: `Generate a brief 3-5 word title for this question, excluding the term 'title.' Then, please reply with only the title: ${userPrompt}`,
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
			const chat = await db.get('chats', _chatId);
			await db.put('chats', { ...chat, title: res.response === '' ? 'New Chat' : res.response });
			if (chat.id === chatId) {
				title = res.response;
			}
		}
	};
</script>

<svelte:window
	on:scroll={(e) => {
		autoScroll = window.innerHeight + window.scrollY >= document.body.offsetHeight - 30;
	}}
/>

<div class="app text-gray-100">
	<div class=" bg-gray-800 min-h-screen overflow-auto flex flex-row">
		<Navbar
			selectedChatId={chatId}
			{chats}
			{title}
			{loadChat}
			{editChatTitle}
			{deleteChat}
			{createNewChat}
			{importChatHistory}
			{exportChatHistory}
			{deleteChatHistory}
			{openSettings}
		/>

		<SettingsModal bind:show={showSettings} {saveSettings} {getModelTags} />

		<div class="min-h-screen w-full flex justify-center">
			<div class=" py-2.5 flex flex-col justify-between w-full">
				<div class="max-w-2xl mx-auto w-full px-2.5 mt-14">
					<div class="p-3 rounded-lg bg-gray-900">
						<div>
							<label
								for="models"
								class="block mb-2 text-sm font-medium text-gray-200 flex justify-between"
							>
								<div class="self-center">Model</div>

								<button
									class=" self-center hover:text-gray-300"
									on:click={() => {
										openSettings();
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
											d="M10.343 3.94c.09-.542.56-.94 1.11-.94h1.093c.55 0 1.02.398 1.11.94l.149.894c.07.424.384.764.78.93.398.164.855.142 1.205-.108l.737-.527a1.125 1.125 0 011.45.12l.773.774c.39.389.44 1.002.12 1.45l-.527.737c-.25.35-.272.806-.107 1.204.165.397.505.71.93.78l.893.15c.543.09.94.56.94 1.109v1.094c0 .55-.397 1.02-.94 1.11l-.893.149c-.425.07-.765.383-.93.78-.165.398-.143.854.107 1.204l.527.738c.32.447.269 1.06-.12 1.45l-.774.773a1.125 1.125 0 01-1.449.12l-.738-.527c-.35-.25-.806-.272-1.203-.107-.397.165-.71.505-.781.929l-.149.894c-.09.542-.56.94-1.11.94h-1.094c-.55 0-1.019-.398-1.11-.94l-.148-.894c-.071-.424-.384-.764-.781-.93-.398-.164-.854-.142-1.204.108l-.738.527c-.447.32-1.06.269-1.45-.12l-.773-.774a1.125 1.125 0 01-.12-1.45l.527-.737c.25-.35.273-.806.108-1.204-.165-.397-.505-.71-.93-.78l-.894-.15c-.542-.09-.94-.56-.94-1.109v-1.094c0-.55.398-1.02.94-1.11l.894-.149c.424-.07.765-.383.93-.78.165-.398.143-.854-.107-1.204l-.527-.738a1.125 1.125 0 01.12-1.45l.773-.773a1.125 1.125 0 011.45-.12l.737.527c.35.25.807.272 1.204.107.397-.165.71-.505.78-.929l.15-.894z"
										/>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
										/>
									</svg>
								</button>
							</label>

							<div>
								<select
									id="models"
									class="outline-none border border-gray-600 bg-gray-700 text-gray-200 text-sm rounded-lg block w-full p-2.5 placeholder-gray-400"
									bind:value={selectedModel}
									disabled={messages.length != 0}
								>
									<option value="" selected>Select a model</option>

									{#each models as model}
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
								<div class="flex justify-between p-5 py-10 max-w-3xl mx-auto rounded-lg group">
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
											<div
												class="prose chat-{message.role} w-full max-w-full prose-invert prose-headings:my-0 prose-p:my-0 prose-pre:my-0 prose-table:my-0 prose-blockquote:my-0 prose-img:my-0 prose-ul:-my-4 prose-ol:-my-4 prose-li:-my-3 prose-ul:-mb-8 prose-ol:-mb-8 prose-li:-mb-4 whitespace-pre-line"
											>
												{#if message.role == 'user'}
													{#if message?.edit === true}
														<div>
															<textarea
																class=" bg-transparent outline-none w-full resize-none"
																bind:value={message.editedContent}
																on:input={(e) => {
																	e.target.style.height = '';
																	e.target.style.height = `${e.target.scrollHeight}px`;
																}}
																on:focus={(e) => {
																	e.target.style.height = '';
																	e.target.style.height = `${e.target.scrollHeight}px`;
																}}
															/>

															<div class=" flex justify-end space-x-2 text-sm text-gray-100">
																<button
																	class="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 transition rounded-lg"
																	on:click={() => {
																		confirmEditMessage(messageIdx);
																	}}
																>
																	Save & Submit
																</button>

																<button
																	class=" px-4 py-2.5 bg-gray-800 hover:bg-gray-700 transition outline outline-1 outline-gray-600 rounded-lg"
																	on:click={() => {
																		cancelEditMessage(messageIdx);
																	}}
																>
																	Cancel
																</button>
															</div>
														</div>
													{:else}
														{message.content}
													{/if}
												{:else}
													{@html marked.parse(message.content)}

													{#if message.done}
														<div class=" flex justify-end space-x-1 text-gray-400">
															<button
																class="p-1 rounded hover:bg-gray-800 {message.rating === 1
																	? 'bg-gray-800'
																	: ''} transition"
																on:click={() => {
																	rateMessage(messageIdx, 1);
																}}
															>
																<svg
																	stroke="currentColor"
																	fill="none"
																	stroke-width="2"
																	viewBox="0 0 24 24"
																	stroke-linecap="round"
																	stroke-linejoin="round"
																	class="w-4 h-4"
																	xmlns="http://www.w3.org/2000/svg"
																	><path
																		d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"
																	/></svg
																>
															</button>
															<button
																class="p-1 rounded hover:bg-gray-800 {message.rating === -1
																	? 'bg-gray-800'
																	: ''} transition"
																on:click={() => {
																	rateMessage(messageIdx, -1);
																}}
															>
																<svg
																	stroke="currentColor"
																	fill="none"
																	stroke-width="2"
																	viewBox="0 0 24 24"
																	stroke-linecap="round"
																	stroke-linejoin="round"
																	class="w-4 h-4"
																	xmlns="http://www.w3.org/2000/svg"
																	><path
																		d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"
																	/></svg
																>
															</button>
														</div>
													{/if}
												{/if}
											</div>
										{/if}
										<!-- {} -->
									</div>

									<div>
										{#if message.role == 'user'}
											{#if message?.edit !== true}
												<button
													class="invisible group-hover:visible p-1 rounded hover:bg-gray-700 transition"
													on:click={() => {
														editMessage(messageIdx);
													}}
												>
													<svg
														xmlns="http://www.w3.org/2000/svg"
														viewBox="0 0 20 20"
														fill="currentColor"
														class="w-4 h-4"
													>
														<path
															d="M5.433 13.917l1.262-3.155A4 4 0 017.58 9.42l6.92-6.918a2.121 2.121 0 013 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 01-.65-.65z"
														/>
														<path
															d="M3.5 5.75c0-.69.56-1.25 1.25-1.25H10A.75.75 0 0010 3H4.75A2.75 2.75 0 002 5.75v9.5A2.75 2.75 0 004.75 18h9.5A2.75 2.75 0 0017 15.25V10a.75.75 0 00-1.5 0v5.25c0 .69-.56 1.25-1.25 1.25h-9.5c-.69 0-1.25-.56-1.25-1.25v-9.5z"
														/>
													</svg>
												</button>
											{/if}
										{:else if message.done}
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
						{#if messages.length == 0 && suggestions !== 'false'}
							<Suggestions {submitPrompt} />
						{/if}

						{#if messages.length != 0 && messages.at(-1).role == 'assistant'}
							{#if messages.at(-1).done == true}
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
							{:else}
								<div class=" flex justify-end mb-2.5">
									<button
										class=" flex px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg"
										on:click={stopResponse}
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
													d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm5-2.25A.75.75 0 017.75 7h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-.75.75h-4.5a.75.75 0 01-.75-.75v-4.5z"
													clip-rule="evenodd"
												/>
											</svg>
										</div>
										<div class=" self-center text-sm">Stop generating</div>
									</button>
								</div>
							{/if}
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
											disabled={prompt === ''}
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

		<div class=" hidden katex" />

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

	pre[class*='language-'] {
		position: relative;
		overflow: auto;

		/* make space  */
		margin: 5px 0;
		padding: 1.75rem 0 1.75rem 1rem;
		border-radius: 10px;
	}

	pre[class*='language-'] button {
		position: absolute;
		top: 5px;
		right: 5px;

		font-size: 0.9rem;
		padding: 0.15rem;
		background-color: #828282;

		border: ridge 1px #7b7b7c;
		border-radius: 5px;
		text-shadow: #c4c4c4 0 0 2px;
	}

	pre[class*='language-'] button:hover {
		cursor: pointer;
		background-color: #bcbabb;
	}
</style>
