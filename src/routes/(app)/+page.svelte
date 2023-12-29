<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';
	import toast from 'svelte-french-toast';

	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';

	import { models, modelfiles, user, settings, db, chats, chatId } from '$lib/stores';
	import { OLLAMA_API_BASE_URL } from '$lib/constants';

	import { generateChatCompletion, generateTitle } from '$lib/apis/ollama';
	import { copyToClipboard, splitStream } from '$lib/utils';

	import MessageInput from '$lib/components/chat/MessageInput.svelte';
	import Messages from '$lib/components/chat/Messages.svelte';
	import ModelSelector from '$lib/components/chat/ModelSelector.svelte';
	import Navbar from '$lib/components/layout/Navbar.svelte';
	import { createNewChat, getChatList, updateChatById } from '$lib/apis/chats';

	let stopResponseFlag = false;
	let autoScroll = true;

	let selectedModels = [''];

	let selectedModelfile = null;
	$: selectedModelfile =
		selectedModels.length === 1 &&
		$modelfiles.filter((modelfile) => modelfile.tagName === selectedModels[0]).length > 0
			? $modelfiles.filter((modelfile) => modelfile.tagName === selectedModels[0])[0]
			: null;

	let chat = null;

	let title = '';
	let prompt = '';
	let files = [];
	let messages = [];
	let history = {
		messages: {},
		currentId: null
	};

	$: if (history.currentId !== null) {
		let _messages = [];

		let currentMessage = history.messages[history.currentId];
		while (currentMessage !== null) {
			_messages.unshift({ ...currentMessage });
			currentMessage =
				currentMessage.parentId !== null ? history.messages[currentMessage.parentId] : null;
		}
		messages = _messages;
	} else {
		messages = [];
	}

	onMount(async () => {
		await initNewChat();
	});

	//////////////////////////
	// Web functions
	//////////////////////////

	const initNewChat = async () => {
		window.history.replaceState(history.state, '', `/`);

		console.log('initNewChat');

		await chatId.set('');
		console.log($chatId);

		autoScroll = true;

		title = '';
		messages = [];
		history = {
			messages: {},
			currentId: null
		};
		selectedModels = $page.url.searchParams.get('models')
			? $page.url.searchParams.get('models')?.split(',')
			: $settings.models ?? [''];

		let _settings = JSON.parse(localStorage.getItem('settings') ?? '{}');
		settings.set({
			..._settings
		});
	};

	//////////////////////////
	// Ollama functions
	//////////////////////////

	const sendPrompt = async (prompt, parentId) => {
		const _chatId = JSON.parse(JSON.stringify($chatId));
		await Promise.all(
			selectedModels.map(async (model) => {
				console.log(model);
				if ($models.filter((m) => m.name === model)[0].external) {
					await sendPromptOpenAI(model, prompt, parentId, _chatId);
				} else {
					await sendPromptOllama(model, prompt, parentId, _chatId);
				}
			})
		);

		await chats.set(await getChatList(localStorage.token));
	};

	const sendPromptOllama = async (model, userPrompt, parentId, _chatId) => {
		// Create response message
		let responseMessageId = uuidv4();
		let responseMessage = {
			parentId: parentId,
			id: responseMessageId,
			childrenIds: [],
			role: 'assistant',
			content: '',
			model: model
		};

		// Add message to history and Set currentId to messageId
		history.messages[responseMessageId] = responseMessage;
		history.currentId = responseMessageId;

		// Append messageId to childrenIds of parent message
		if (parentId !== null) {
			history.messages[parentId].childrenIds = [
				...history.messages[parentId].childrenIds,
				responseMessageId
			];
		}

		// Wait until history/message have been updated
		await tick();

		// Scroll down
		window.scrollTo({ top: document.body.scrollHeight });

		const res = await generateChatCompletion(
			$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL,
			localStorage.token,
			{
				model: model,
				messages: [
					$settings.system
						? {
								role: 'system',
								content: $settings.system
						  }
						: undefined,
					...messages
				]
					.filter((message) => message)
					.map((message) => ({
						role: message.role,
						content: message.content,
						...(message.files && {
							images: message.files
								.filter((file) => file.type === 'image')
								.map((file) => file.url.slice(file.url.indexOf(',') + 1))
						})
					})),
				options: {
					...($settings.options ?? {})
				},
				format: $settings.requestFormat ?? undefined
			}
		);

		if (res && res.ok) {
			const reader = res.body
				.pipeThrough(new TextDecoderStream())
				.pipeThrough(splitStream('\n'))
				.getReader();

			while (true) {
				const { value, done } = await reader.read();
				if (done || stopResponseFlag || _chatId !== $chatId) {
					responseMessage.done = true;
					messages = messages;
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
								if (responseMessage.content == '' && data.message.content == '\n') {
									continue;
								} else {
									responseMessage.content += data.message.content;
									messages = messages;
								}
							} else {
								responseMessage.done = true;

								if (responseMessage.content == '') {
									responseMessage.error = true;
									responseMessage.content =
										'Oops! No text generated from Ollama, Please try again.';
								}

								responseMessage.context = data.context ?? null;
								responseMessage.info = {
									total_duration: data.total_duration,
									load_duration: data.load_duration,
									sample_count: data.sample_count,
									sample_duration: data.sample_duration,
									prompt_eval_count: data.prompt_eval_count,
									prompt_eval_duration: data.prompt_eval_duration,
									eval_count: data.eval_count,
									eval_duration: data.eval_duration
								};
								messages = messages;

								if ($settings.notificationEnabled && !document.hasFocus()) {
									const notification = new Notification(
										selectedModelfile
											? `${
													selectedModelfile.title.charAt(0).toUpperCase() +
													selectedModelfile.title.slice(1)
											  }`
											: `Ollama - ${model}`,
										{
											body: responseMessage.content,
											icon: selectedModelfile?.imageUrl ?? '/favicon.png'
										}
									);
								}

								if ($settings.responseAutoCopy) {
									copyToClipboard(responseMessage.content);
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

				if (autoScroll) {
					window.scrollTo({ top: document.body.scrollHeight });
				}
			}

			if ($chatId == _chatId) {
				chat = await updateChatById(localStorage.token, _chatId, {
					messages: messages,
					history: history
				});
				await chats.set(await getChatList(localStorage.token));
			}
		} else {
			if (res !== null) {
				const error = await res.json();
				console.log(error);
				if ('detail' in error) {
					toast.error(error.detail);
					responseMessage.content = error.detail;
				} else {
					toast.error(error.error);
					responseMessage.content = error.error;
				}
			} else {
				toast.error(`Uh-oh! There was an issue connecting to Ollama.`);
				responseMessage.content = `Uh-oh! There was an issue connecting to Ollama.`;
			}

			responseMessage.error = true;
			responseMessage.content = `Uh-oh! There was an issue connecting to Ollama.`;
			responseMessage.done = true;
			messages = messages;
		}

		stopResponseFlag = false;
		await tick();

		if (autoScroll) {
			window.scrollTo({ top: document.body.scrollHeight });
		}

		if (messages.length == 2 && messages.at(1).content !== '') {
			window.history.replaceState(history.state, '', `/c/${_chatId}`);
			await generateChatTitle(_chatId, userPrompt);
		}
	};

	const sendPromptOpenAI = async (model, userPrompt, parentId, _chatId) => {
		if ($settings.OPENAI_API_KEY) {
			if (models) {
				let responseMessageId = uuidv4();

				let responseMessage = {
					parentId: parentId,
					id: responseMessageId,
					childrenIds: [],
					role: 'assistant',
					content: '',
					model: model
				};

				history.messages[responseMessageId] = responseMessage;
				history.currentId = responseMessageId;
				if (parentId !== null) {
					history.messages[parentId].childrenIds = [
						...history.messages[parentId].childrenIds,
						responseMessageId
					];
				}

				window.scrollTo({ top: document.body.scrollHeight });

				const res = await fetch(
					`${$settings.OPENAI_API_BASE_URL ?? 'https://api.openai.com/v1'}/chat/completions`,
					{
						method: 'POST',
						headers: {
							Authorization: `Bearer ${$settings.OPENAI_API_KEY}`,
							'Content-Type': 'application/json'
						},
						body: JSON.stringify({
							model: model,
							stream: true,
							messages: [
								$settings.system
									? {
											role: 'system',
											content: $settings.system
									  }
									: undefined,
								...messages
							]
								.filter((message) => message)
								.map((message) => ({
									role: message.role,
									...(message.files
										? {
												content: [
													{
														type: 'text',
														text: message.content
													},
													...message.files
														.filter((file) => file.type === 'image')
														.map((file) => ({
															type: 'image_url',
															image_url: {
																url: file.url
															}
														}))
												]
										  }
										: { content: message.content })
								})),
							seed: $settings.options.seed ?? undefined,
							stop: $settings.options.stop ?? undefined,
							temperature: $settings.options.temperature ?? undefined,
							top_p: $settings.options.top_p ?? undefined,
							num_ctx: $settings.options.num_ctx ?? undefined,
							frequency_penalty: $settings.options.repeat_penalty ?? undefined,
							max_tokens: $settings.options.num_predict ?? undefined,
						})
					}
				).catch((err) => {
					console.log(err);
					return null;
				});

				if (res && res.ok) {
					const reader = res.body
						.pipeThrough(new TextDecoderStream())
						.pipeThrough(splitStream('\n'))
						.getReader();

					while (true) {
						const { value, done } = await reader.read();
						if (done || stopResponseFlag || _chatId !== $chatId) {
							responseMessage.done = true;
							messages = messages;
							break;
						}

						try {
							let lines = value.split('\n');

							for (const line of lines) {
								if (line !== '') {
									console.log(line);
									if (line === 'data: [DONE]') {
										responseMessage.done = true;
										messages = messages;
									} else {
										let data = JSON.parse(line.replace(/^data: /, ''));
										console.log(data);

										if (responseMessage.content == '' && data.choices[0].delta.content == '\n') {
											continue;
										} else {
											responseMessage.content += data.choices[0].delta.content ?? '';
											messages = messages;
										}
									}
								}
							}
						} catch (error) {
							console.log(error);
						}

						if ($settings.notificationEnabled && !document.hasFocus()) {
							const notification = new Notification(`OpenAI ${model}`, {
								body: responseMessage.content,
								icon: '/favicon.png'
							});
						}

						if ($settings.responseAutoCopy) {
							copyToClipboard(responseMessage.content);
						}

						if (autoScroll) {
							window.scrollTo({ top: document.body.scrollHeight });
						}
					}

					if ($chatId == _chatId) {
						chat = await updateChatById(localStorage.token, _chatId, {
							messages: messages,
							history: history
						});
						await chats.set(await getChatList(localStorage.token));
					}
				} else {
					if (res !== null) {
						const error = await res.json();
						console.log(error);
						if ('detail' in error) {
							toast.error(error.detail);
							responseMessage.content = error.detail;
						} else {
							if ('message' in error.error) {
								toast.error(error.error.message);
								responseMessage.content = error.error.message;
							} else {
								toast.error(error.error);
								responseMessage.content = error.error;
							}
						}
					} else {
						toast.error(`Uh-oh! There was an issue connecting to ${model}.`);
						responseMessage.content = `Uh-oh! There was an issue connecting to ${model}.`;
					}

					responseMessage.error = true;
					responseMessage.content = `Uh-oh! There was an issue connecting to ${model}.`;
					responseMessage.done = true;
					messages = messages;
				}

				stopResponseFlag = false;
				await tick();

				if (autoScroll) {
					window.scrollTo({ top: document.body.scrollHeight });
				}

				if (messages.length == 2) {
					window.history.replaceState(history.state, '', `/c/${_chatId}`);
					await setChatTitle(_chatId, userPrompt);
				}
			}
		}
	};

	const submitPrompt = async (userPrompt) => {
		console.log('submitPrompt', $chatId);

		if (selectedModels.includes('')) {
			toast.error('Model not selected');
		} else if (messages.length != 0 && messages.at(-1).done != true) {
			// Response not done
			console.log('wait');
		} else {
			// Reset chat message textarea height
			document.getElementById('chat-textarea').style.height = '';

			// Create user message
			let userMessageId = uuidv4();
			let userMessage = {
				id: userMessageId,
				parentId: messages.length !== 0 ? messages.at(-1).id : null,
				childrenIds: [],
				role: 'user',
				content: userPrompt,
				files: files.length > 0 ? files : undefined
			};

			// Add message to history and Set currentId to messageId
			history.messages[userMessageId] = userMessage;
			history.currentId = userMessageId;

			// Append messageId to childrenIds of parent message
			if (messages.length !== 0) {
				history.messages[messages.at(-1).id].childrenIds.push(userMessageId);
			}

			// Wait until history/message have been updated
			await tick();

			// Create new chat if only one message in messages
			if (messages.length == 1) {
				chat = await createNewChat(localStorage.token, {
					id: $chatId,
					title: 'New Chat',
					models: selectedModels,
					system: $settings.system ?? undefined,
					options: {
						...($settings.options ?? {})
					},
					messages: messages,
					history: history,
					timestamp: Date.now()
				});
				await chats.set(await getChatList(localStorage.token));
				await chatId.set(chat.id);
				await tick();
			}

			// Reset chat input textarea
			prompt = '';
			files = [];

			// Send prompt
			await sendPrompt(userPrompt, userMessageId);
		}
	};

	const stopResponse = () => {
		stopResponseFlag = true;
		console.log('stopResponse');
	};

	const regenerateResponse = async () => {
		console.log('regenerateResponse');
		if (messages.length != 0 && messages.at(-1).done == true) {
			messages.splice(messages.length - 1, 1);
			messages = messages;

			let userMessage = messages.at(-1);
			let userPrompt = userMessage.content;

			await sendPrompt(userPrompt, userMessage.id);
		}
	};

	const generateChatTitle = async (_chatId, userPrompt) => {
		if ($settings.titleAutoGenerate ?? true) {
			const title = await generateTitle(
				$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL,
				localStorage.token,
				selectedModels[0],
				userPrompt
			);

			if (title) {
				await setChatTitle(_chatId, title);
			}
		} else {
			await setChatTitle(_chatId, `${userPrompt}`);
		}
	};

	const setChatTitle = async (_chatId, _title) => {
		if (_chatId === $chatId) {
			title = _title;
		}

		chat = await updateChatById(localStorage.token, _chatId, { title: _title });
		await chats.set(await getChatList(localStorage.token));
	};
</script>

<svelte:window
	on:scroll={(e) => {
		autoScroll = window.innerHeight + window.scrollY >= document.body.offsetHeight - 40;
	}}
/>

<Navbar {title} shareEnabled={messages.length > 0} {initNewChat} />
<div class="min-h-screen w-full flex justify-center">
	<div class=" py-2.5 flex flex-col justify-between w-full">
		<div class="max-w-2xl mx-auto w-full px-3 md:px-0 mt-10">
			<ModelSelector bind:selectedModels disabled={messages.length > 0} />
		</div>

		<div class=" h-full mt-10 mb-32 w-full flex flex-col">
			<Messages
				chatId={$chatId}
				{selectedModels}
				{selectedModelfile}
				bind:history
				bind:messages
				bind:autoScroll
				bottomPadding={files.length > 0}
				{sendPrompt}
				{regenerateResponse}
			/>
		</div>
	</div>

	<MessageInput
		bind:files
		bind:prompt
		bind:autoScroll
		suggestionPrompts={selectedModelfile?.suggestionPrompts ?? [
			{
				title: ['Help me study', 'vocabulary for a college entrance exam'],
				content: `Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option.`
			},
			{
				title: ['Give me ideas', `for what to do with my kids' art`],
				content: `What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.`
			},
			{
				title: ['Tell me a fun fact', 'about the Roman Empire'],
				content: 'Tell me a random fun fact about the Roman Empire'
			},
			{
				title: ['Show me a code snippet', `of a website's sticky header`],
				content: `Show me a code snippet of a website's sticky header in CSS and JavaScript.`
			}
		]}
		{messages}
		{submitPrompt}
		{stopResponse}
	/>
</div>
