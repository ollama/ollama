<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';
	import toast from 'svelte-french-toast';

	import { OLLAMA_API_BASE_URL } from '$lib/constants';
	import { onMount, tick } from 'svelte';
	import { splitStream } from '$lib/utils';
	import { goto } from '$app/navigation';

	import { config, user, settings, db, chats, chatId } from '$lib/stores';

	import MessageInput from '$lib/components/chat/MessageInput.svelte';
	import Messages from '$lib/components/chat/Messages.svelte';
	import ModelSelector from '$lib/components/chat/ModelSelector.svelte';
	import Navbar from '$lib/components/layout/Navbar.svelte';

	let stopResponseFlag = false;
	let autoScroll = true;

	let selectedModels = [''];

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
	}

	onMount(async () => {
		await chatId.set(uuidv4());

		chatId.subscribe(async () => {
			await initNewChat();
		});
	});

	//////////////////////////
	// Web functions
	//////////////////////////

	const initNewChat = async () => {
		console.log($chatId);

		autoScroll = true;

		title = '';
		messages = [];
		history = {
			messages: {},
			currentId: null
		};
		selectedModels = $settings.models ?? [''];
	};

	//////////////////////////
	// Ollama functions
	//////////////////////////

	const sendPrompt = async (userPrompt, parentId) => {
		await Promise.all(
			selectedModels.map(async (model) => {
				if (model.includes('gpt-')) {
					await sendPromptOpenAI(model, userPrompt, parentId);
				} else {
					await sendPromptOllama(model, userPrompt, parentId);
				}
			})
		);

		await chats.set(await $db.getChats());
	};

	const sendPromptOllama = async (model, userPrompt, parentId) => {
		console.log('sendPromptOllama');
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

		const res = await fetch(`${$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream',
				...($settings.authHeader && { Authorization: $settings.authHeader }),
				...($user && { Authorization: `Bearer ${localStorage.token}` })
			},
			body: JSON.stringify({
				model: model,
				prompt: userPrompt,
				system: $settings.system ?? undefined,
				options: {
					seed: $settings.seed ?? undefined,
					temperature: $settings.temperature ?? undefined,
					repeat_penalty: $settings.repeat_penalty ?? undefined,
					top_k: $settings.top_k ?? undefined,
					top_p: $settings.top_p ?? undefined,
					num_ctx:  $settings.num_ctx ?? undefined
				},
				format: $settings.requestFormat ?? undefined,
				context:
					history.messages[parentId] !== null &&
					history.messages[parentId].parentId in history.messages
						? history.messages[history.messages[parentId].parentId]?.context ?? undefined
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
						} else if ('detail' in data) {
							throw data;
						} else {
							responseMessage.done = true;
							responseMessage.context = data.context;
							messages = messages;
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

			await $db.updateChatById($chatId, {
				title: title === '' ? 'New Chat' : title,
				models: selectedModels,
				system: $settings.system ?? undefined,
				options: {
					seed: $settings.seed ?? undefined,
					temperature: $settings.temperature ?? undefined,
					repeat_penalty: $settings.repeat_penalty ?? undefined,
					top_k: $settings.top_k ?? undefined,
					top_p: $settings.top_p ?? undefined,
					num_ctx:  $settings.num_ctx ?? undefined
				},
				messages: messages,
				history: history
			});
		}

		stopResponseFlag = false;
		await tick();
		if (autoScroll) {
			window.scrollTo({ top: document.body.scrollHeight });
		}

		if (messages.length == 2 && messages.at(1).content !== '') {
			window.history.replaceState(history.state, '', `/c/${$chatId}`);
			await generateChatTitle($chatId, userPrompt);
		}
	};

	const sendPromptOpenAI = async (model, userPrompt, parentId) => {
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

				await tick();

				window.scrollTo({ top: document.body.scrollHeight });

				const res = await fetch(`https://api.openai.com/v1/chat/completions`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${$settings.OPENAI_API_KEY}`
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
							.map((message) => ({ role: message.role, content: message.content })),
						temperature: $settings.temperature ?? undefined,
						top_p: $settings.top_p ?? undefined,
						num_ctx:  $settings.num_ctx ?? undefined,
						frequency_penalty: $settings.repeat_penalty ?? undefined
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
						}

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

					if (autoScroll) {
						window.scrollTo({ top: document.body.scrollHeight });
					}

					await $db.updateChatById($chatId, {
						title: title === '' ? 'New Chat' : title,
						models: selectedModels,
						system: $settings.system ?? undefined,
						options: {
							seed: $settings.seed ?? undefined,
							temperature: $settings.temperature ?? undefined,
							repeat_penalty: $settings.repeat_penalty ?? undefined,
							top_k: $settings.top_k ?? undefined,
							top_p: $settings.top_p ?? undefined,
							num_ctx:  $settings.num_ctx ?? undefined
						},
						messages: messages,
						history: history
					});
				}

				stopResponseFlag = false;

				await tick();
				if (autoScroll) {
					window.scrollTo({ top: document.body.scrollHeight });
				}

				if (messages.length == 2) {
					window.history.replaceState(history.state, '', `/c/${$chatId}`);
					await setChatTitle($chatId, userPrompt);
				}
			}
		}
	};

	const submitPrompt = async (userPrompt) => {
		console.log('submitPrompt');

		if (selectedModels.includes('')) {
			toast.error('Model not selected');
		} else if (messages.length != 0 && messages.at(-1).done != true) {
			console.log('wait');
		} else {
			document.getElementById('chat-textarea').style.height = '';

			let userMessageId = uuidv4();
			let userMessage = {
				id: userMessageId,
				parentId: messages.length !== 0 ? messages.at(-1).id : null,
				childrenIds: [],
				role: 'user',
				content: userPrompt,
				files: files.length > 0 ? files : undefined
			};

			if (messages.length !== 0) {
				history.messages[messages.at(-1).id].childrenIds.push(userMessageId);
			}

			history.messages[userMessageId] = userMessage;
			history.currentId = userMessageId;

			prompt = '';
			files = [];

			if (messages.length == 0) {
				await $db.createNewChat({
					id: $chatId,
					title: 'New Chat',
					models: selectedModels,
					system: $settings.system ?? undefined,
					options: {
						seed: $settings.seed ?? undefined,
						temperature: $settings.temperature ?? undefined,
						repeat_penalty: $settings.repeat_penalty ?? undefined,
						top_k: $settings.top_k ?? undefined,
						top_p: $settings.top_p ?? undefined,
						num_ctx:  $settings.num_ctx ?? undefined
					},
					messages: messages,
					history: history
				});
			}

			setTimeout(() => {
				window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
			}, 50);

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
			console.log('generateChatTitle');

			const res = await fetch(`${$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/generate`, {
				method: 'POST',
				headers: {
					'Content-Type': 'text/event-stream',
					...($settings.authHeader && { Authorization: $settings.authHeader }),
					...($user && { Authorization: `Bearer ${localStorage.token}` })
				},
				body: JSON.stringify({
					model: selectedModels[0],
					prompt: `Generate a brief 3-5 word title for this question, excluding the term 'title.' Then, please reply with only the title: ${userPrompt}`,
					stream: false
				})
			})
				.then(async (res) => {
					if (!res.ok) throw await res.json();
					return res.json();
				})
				.catch((error) => {
					if ('detail' in error) {
						toast.error(error.detail);
					}
					console.log(error);
					return null;
				});

			if (res) {
				await setChatTitle(_chatId, res.response === '' ? 'New Chat' : res.response);
			}
		} else {
			await setChatTitle(_chatId, `${userPrompt}`);
		}
	};

	const setChatTitle = async (_chatId, _title) => {
		await $db.updateChatById(_chatId, { title: _title });
		if (_chatId === $chatId) {
			title = _title;
		}
	};
</script>

<svelte:window
	on:scroll={(e) => {
		autoScroll = window.innerHeight + window.scrollY >= document.body.offsetHeight - 40;
	}}
/>

<Navbar {title} />
<div class="min-h-screen w-full flex justify-center">
	<div class=" py-2.5 flex flex-col justify-between w-full">
		<div class="max-w-2xl mx-auto w-full px-3 md:px-0 mt-10">
			<ModelSelector bind:selectedModels disabled={messages.length > 0} />
		</div>

		<div class=" h-full mt-10 mb-32 w-full flex flex-col">
			<Messages bind:history bind:messages bind:autoScroll {sendPrompt} {regenerateResponse} />
		</div>
	</div>

	<MessageInput bind:prompt bind:files bind:autoScroll {messages} {submitPrompt} {stopResponse} />
</div>
