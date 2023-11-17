<script lang="ts">
	import { openDB, deleteDB } from 'idb';
	import { v4 as uuidv4 } from 'uuid';
	import { marked } from 'marked';
	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;
	import hljs from 'highlight.js';
	import 'highlight.js/styles/github-dark.min.css';
	import auto_render from 'katex/dist/contrib/auto-render.mjs';
	import 'katex/dist/katex.min.css';
	import toast from 'svelte-french-toast';

	import { API_BASE_URL as BUILD_TIME_API_BASE_URL } from '$lib/constants';
	import { onMount, tick } from 'svelte';

	import Navbar from '$lib/components/layout/Navbar.svelte';
	import SettingsModal from '$lib/components/chat/SettingsModal.svelte';
	import Suggestions from '$lib/components/chat/Suggestions.svelte';

	let API_BASE_URL = BUILD_TIME_API_BASE_URL;
	let db;

	// let selectedModel = '';
	let selectedModels = [''];
	let settings = {
		system: null,
		temperature: null
	};

	let fileUploadEnabled = false;

	let speechRecognition;
	let speechRecognitionEnabled = true;
	let speechRecognitionListening = false;

	let models = [];
	let chats = [];

	let chatId = uuidv4();
	let title = '';
	let prompt = '';
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

	let showSettings = false;
	let stopResponseFlag = false;
	let autoScroll = true;
	let suggestions = ''; // $page.url.searchParams.get('suggestions');

	onMount(async () => {
		await Promise.all([await createNewChat(true), await setDBandLoadChats()]);
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
				let code = block.querySelector('code');
				code.style.borderTopRightRadius = 0;
				code.style.borderTopLeftRadius = 0;

				let topBarDiv = document.createElement('div');
				topBarDiv.style.backgroundColor = '#202123';
				topBarDiv.style.overflowX = 'auto';
				topBarDiv.style.display = 'flex';
				topBarDiv.style.justifyContent = 'space-between';
				topBarDiv.style.padding = '0 1rem';
				topBarDiv.style.paddingTop = '4px';
				topBarDiv.style.borderTopRightRadius = '8px';
				topBarDiv.style.borderTopLeftRadius = '8px';

				let langDiv = document.createElement('div');

				let codeClassNames = code?.className.split(' ');
				langDiv.textContent =
					codeClassNames[0] === 'hljs' ? codeClassNames[1].slice(9) : codeClassNames[0].slice(9);
				langDiv.style.color = 'white';
				langDiv.style.margin = '4px';
				langDiv.style.fontSize = '0.75rem';

				let button = document.createElement('button');
				button.textContent = 'Copy Code';
				button.style.background = 'none';
				button.style.fontSize = '0.75rem';
				button.style.border = 'none';
				button.style.margin = '4px';
				button.style.cursor = 'pointer';
				button.style.color = '#ddd';
				button.addEventListener('click', () => copyCode(block, button));

				topBarDiv.appendChild(langDiv);
				topBarDiv.appendChild(button);

				block.prepend(topBarDiv);

				// button.addEventListener('click', async () => {
				// 	await copyCode(block, button);
				// });
			}
		});

		async function copyCode(block, button) {
			let code = block.querySelector('code');
			let text = code.innerText;

			await navigator.clipboard.writeText(text);

			// visual feedback that task is completed
			button.innerText = 'Copied!';

			setTimeout(() => {
				button.innerText = 'Copy Code';
			}, 1000);
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
					{ left: '$', right: '$', display: true },
					{ left: '\\(', right: '\\)', display: true },
					{ left: '\\[', right: '\\]', display: true }
				],
				// • rendering keys, e.g.:
				throwOnError: false
			});
		}
	};

	const speechRecognitionHandler = () => {
		// Check if SpeechRecognition is supported

		if (speechRecognitionListening) {
			speechRecognition.stop();
		} else {
			if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
				// Create a SpeechRecognition object
				speechRecognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();

				// Set continuous to true for continuous recognition
				speechRecognition.continuous = true;

				// Set the timeout for turning off the recognition after inactivity (in milliseconds)
				const inactivityTimeout = 3000; // 3 seconds

				let timeoutId;
				// Start recognition
				speechRecognition.start();
				speechRecognitionListening = true;

				// Event triggered when speech is recognized
				speechRecognition.onresult = function (event) {
					// Clear the inactivity timeout
					clearTimeout(timeoutId);

					// Handle recognized speech
					console.log(event);
					const transcript = event.results[Object.keys(event.results).length - 1][0].transcript;
					prompt = `${prompt}${transcript}`;

					// Restart the inactivity timeout
					timeoutId = setTimeout(() => {
						console.log('Speech recognition turned off due to inactivity.');
						speechRecognition.stop();
					}, inactivityTimeout);
				};

				// Event triggered when recognition is ended
				speechRecognition.onend = function () {
					// Restart recognition after it ends
					console.log('recognition ended');
					speechRecognitionListening = false;
					if (prompt !== '' && settings?.speechAutoSend === true) {
						submitPrompt(prompt);
					}
				};

				// Event triggered when an error occurs
				speechRecognition.onerror = function (event) {
					console.log(event);
					toast.error(`Speech recognition error: ${event.error}`);
					speechRecognitionListening = false;
				};
			} else {
				toast.error('SpeechRecognition API is not supported in this browser.');
			}
		}
	};

	//////////////////////////
	// Web functions
	//////////////////////////

	const createNewChat = async (init = false) => {
		if (init || messages.length > 0) {
			chatId = uuidv4();
			autoScroll = true;

			title = '';
			messages = [];
			history = {
				messages: {},
				currentId: null
			};

			settings = JSON.parse(localStorage.getItem('settings') ?? JSON.stringify(settings));

			API_BASE_URL = settings?.API_BASE_URL ?? BUILD_TIME_API_BASE_URL;
			console.log(API_BASE_URL);

			if (models.length === 0) {
				await getModelTags();
			}

			// selectedModel =
			// 	settings.model && models.map((model) => model.name).includes(settings.model)
			// 		? settings.model
			// 		: '';

			selectedModels = settings.models ?? [''];

			console.log(chatId);
		}
	};

	const setDBandLoadChats = async () => {
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
	};

	const saveDefaultModel = () => {
		settings.models = selectedModels;
		localStorage.setItem('settings', JSON.stringify(settings));
		toast.success('Default model updated');
	};

	const saveSettings = async (updated) => {
		console.log(updated);
		settings = { ...settings, ...updated };
		localStorage.setItem('settings', JSON.stringify(settings));
		API_BASE_URL = updated?.API_BASE_URL ?? API_BASE_URL;
		await getModelTags();
	};

	const loadChat = async (id) => {
		const chat = await db.get('chats', id);
		console.log(chat);
		if (chatId !== chat.id) {
			if ('history' in chat && chat.history !== undefined) {
				history = chat.history;
			} else {
				let _history = {
					messages: {},
					currentId: null
				};

				let parentMessageId = null;
				let messageId = null;

				for (const message of chat.messages) {
					messageId = uuidv4();

					if (parentMessageId !== null) {
						_history.messages[parentMessageId].childrenIds = [
							..._history.messages[parentMessageId].childrenIds,
							messageId
						];
					}

					_history.messages[messageId] = {
						...message,
						id: messageId,
						parentId: parentMessageId,
						childrenIds: []
					};

					parentMessageId = messageId;
				}
				_history.currentId = messageId;

				history = _history;
			}

			if ('models' in chat && chat.models !== undefined) {
				selectedModels = chat.models ?? selectedModels;
			} else {
				selectedModels = [chat.model ?? ''];
			}

			console.log(history);

			title = chat.title;
			chatId = chat.id;
			settings.system = chat.system ?? settings.system;
			settings.temperature = chat.temperature ?? settings.temperature;
			autoScroll = true;

			await tick();

			if (messages.length > 0) {
				history.messages[messages.at(-1).id].done = true;
			}

			renderLatex();
			hljs.highlightAll();
			createCopyCodeBlockButton();
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

	const importChatHistory = async (chatHistory) => {
		for (const chat of chatHistory) {
			console.log(chat);

			await db.put('chats', {
				id: chat.id,
				model: chat.model,
				models: chat.models,
				system: chat.system,
				options: chat.options,
				title: chat.title,
				timestamp: chat.timestamp,
				messages: chat.messages,
				history: chat.history
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

	const editMessageHandler = async (messageId) => {
		// let editMessage = history.messages[messageId];
		history.messages[messageId].edit = true;
		history.messages[messageId].editedContent = history.messages[messageId].content;
	};

	const confirmEditMessage = async (messageId) => {
		history.messages[messageId].edit = false;

		let userPrompt = history.messages[messageId].editedContent;
		let userMessageId = uuidv4();

		let userMessage = {
			id: userMessageId,
			parentId: history.messages[messageId].parentId,
			childrenIds: [],
			role: 'user',
			content: userPrompt
		};

		let messageParentId = history.messages[messageId].parentId;

		if (messageParentId !== null) {
			history.messages[messageParentId].childrenIds = [
				...history.messages[messageParentId].childrenIds,
				userMessageId
			];
		}

		history.messages[userMessageId] = userMessage;
		history.currentId = userMessageId;

		await tick();
		await sendPrompt(userPrompt, userMessageId);
	};

	const cancelEditMessage = (messageId) => {
		history.messages[messageId].edit = false;
		history.messages[messageId].editedContent = undefined;
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
			models: selectedModels,
			system: settings.system,
			options: {
				temperature: settings.temperature
			},
			timestamp: Date.now(),
			messages: messages,
			history: history
		});

		console.log(messages);
	};

	const showPreviousMessage = async (message) => {
		if (message.parentId !== null) {
			let messageId =
				history.messages[message.parentId].childrenIds[
					Math.max(history.messages[message.parentId].childrenIds.indexOf(message.id) - 1, 0)
				];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		} else {
			let childrenIds = Object.values(history.messages)
				.filter((message) => message.parentId === null)
				.map((message) => message.id);
			let messageId = childrenIds[Math.max(childrenIds.indexOf(message.id) - 1, 0)];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		}

		await tick();

		renderLatex();
		hljs.highlightAll();
		createCopyCodeBlockButton();
	};

	const showNextMessage = async (message) => {
		if (message.parentId !== null) {
			let messageId =
				history.messages[message.parentId].childrenIds[
					Math.min(
						history.messages[message.parentId].childrenIds.indexOf(message.id) + 1,
						history.messages[message.parentId].childrenIds.length - 1
					)
				];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		} else {
			let childrenIds = Object.values(history.messages)
				.filter((message) => message.parentId === null)
				.map((message) => message.id);
			let messageId =
				childrenIds[Math.min(childrenIds.indexOf(message.id) + 1, childrenIds.length - 1)];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		}

		await tick();

		renderLatex();
		hljs.highlightAll();
		createCopyCodeBlockButton();
	};

	//////////////////////////
	// Ollama functions
	//////////////////////////

	const getModelTags = async (url = null, type = 'all') => {
		const res = await fetch(`${url === null ? API_BASE_URL : url}/tags`, {
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				...(settings.authHeader && { Authorization: settings.authHeader })
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

		if (type === 'all') {
			if (settings.OPENAI_API_KEY) {
				// Validate OPENAI_API_KEY
				const openaiModelRes = await fetch(`https://api.openai.com/v1/models`, {
					method: 'GET',
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${settings.OPENAI_API_KEY}`
					}
				})
					.then(async (res) => {
						if (!res.ok) throw await res.json();
						return res.json();
					})
					.catch((error) => {
						console.log(error);
						toast.error(`OpenAI: ${error?.error?.message ?? 'Network Problem'}`);
						return null;
					});
				const openaiModels = openaiModelRes?.data ?? null;

				if (openaiModels) {
					models = [
						...(res?.models ?? []),
						{ name: 'hr' },

						...openaiModels
							.map((model) => ({ name: model.id, label: 'OpenAI' }))
							.filter((model) => model.name.includes('gpt'))
					];
				} else {
					models = res?.models ?? [];
				}
			} else {
				models = res?.models ?? [];
			}

			return models;
		} else {
			return res?.models ?? null;
		}
	};

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

		// if (selectedModel.includes('gpt-')) {
		// 	await sendPromptOpenAI(userPrompt, parentId);
		// } else {
		// 	await sendPromptOllama(userPrompt, parentId);
		// }

		console.log(history);
	};

	const sendPromptOllama = async (model, userPrompt, parentId) => {
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

		const res = await fetch(`${API_BASE_URL}/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream',
				...(settings.authHeader && { Authorization: settings.authHeader })
			},
			body: JSON.stringify({
				model: model,
				prompt: userPrompt,
				system: settings.system ?? undefined,
				options: {
					seed: settings.seed ?? undefined,
					temperature: settings.temperature ?? undefined,
					repeat_penalty: settings.repeat_penalty ?? undefined,
					top_k: settings.top_k ?? undefined,
					top_p: settings.top_p ?? undefined
				},
				format: settings.requestFormat ?? undefined,
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
				models: selectedModels,
				system: settings.system,
				options: {
					temperature: settings.temperature
				},
				timestamp: Date.now(),
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
			await generateChatTitle(chatId, userPrompt);
		}
	};

	const sendPromptOpenAI = async (model, userPrompt, parentId) => {
		if (settings.OPENAI_API_KEY) {
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

				const res = await fetch(`https://api.openai.com/v1/chat/completions`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${settings.OPENAI_API_KEY}`
					},
					body: JSON.stringify({
						model: model,
						stream: true,
						messages: [
							settings.system
								? {
										role: 'system',
										content: settings.system
								  }
								: undefined,
							...messages
						]
							.filter((message) => message)
							.map((message) => ({ role: message.role, content: message.content })),
						temperature: settings.temperature ?? undefined,
						top_p: settings.top_p ?? undefined,
						frequency_penalty: settings.repeat_penalty ?? undefined
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

					await db.put('chats', {
						id: chatId,
						title: title === '' ? 'New Chat' : title,
						models: selectedModels,

						system: settings.system,
						options: {
							temperature: settings.temperature
						},
						timestamp: Date.now(),
						messages: messages,
						history: history
					});
				}

				stopResponseFlag = false;

				hljs.highlightAll();
				createCopyCodeBlockButton();
				renderLatex();

				await tick();
				if (autoScroll) {
					window.scrollTo({ top: document.body.scrollHeight });
				}

				if (messages.length == 2) {
					await setChatTitle(chatId, userPrompt);
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
				content: userPrompt
			};

			if (messages.length !== 0) {
				history.messages[messages.at(-1).id].childrenIds.push(userMessageId);
			}

			history.messages[userMessageId] = userMessage;
			history.currentId = userMessageId;

			prompt = '';

			if (messages.length == 0) {
				await db.put('chats', {
					id: chatId,
					models: selectedModels,
					system: settings.system,
					options: {
						temperature: settings.temperature
					},
					title: 'New Chat',
					timestamp: Date.now(),
					messages: messages,
					history: history
				});
				chats = await db.getAllFromIndex('chats', 'timestamp');
			}

			setTimeout(() => {
				window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
			}, 50);

			await sendPrompt(userPrompt, userMessageId);

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

			await sendPrompt(userPrompt, userMessage.id);

			chats = await db.getAllFromIndex('chats', 'timestamp');
		}
	};

	const stopResponse = () => {
		stopResponseFlag = true;
		console.log('stopResponse');
	};

	const generateChatTitle = async (_chatId, userPrompt) => {
		console.log('generateChatTitle');

		const res = await fetch(`${API_BASE_URL}/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream',
				...(settings.authHeader && { Authorization: settings.authHeader })
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
				console.log(error);
				return null;
			});

		if (res) {
			await setChatTitle(_chatId, res.response === '' ? 'New Chat' : res.response);
		}
	};

	const setChatTitle = async (_chatId, _title) => {
		const chat = await db.get('chats', _chatId);
		await db.put('chats', { ...chat, title: _title });
		if (chat.id === chatId) {
			title = _title;
		}
	};
</script>

<svelte:window
	on:scroll={(e) => {
		autoScroll = window.innerHeight + window.scrollY >= document.body.offsetHeight - 40;
	}}
/>

<div class="app">
	<div
		class=" text-gray-700 dark:text-gray-100 bg-white dark:bg-gray-800 min-h-screen overflow-auto flex flex-row"
	>
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
				<div class="max-w-2xl mx-auto w-full px-3 md:px-0 mt-10">
					<div class="flex flex-col my-2">
						{#each selectedModels as selectedModel, selectedModelIdx}
							<div class="flex">
								<select
									id="models"
									class="outline-none bg-transparent text-lg font-semibold rounded-lg block w-full placeholder-gray-400"
									bind:value={selectedModel}
									disabled={messages.length != 0}
								>
									<option class=" text-gray-700" value="" selected>Select a model</option>

									{#each models as model}
										{#if model.name === 'hr'}
											<hr />
										{:else}
											<option value={model.name} class="text-gray-700 text-lg">{model.name}</option>
										{/if}
									{/each}
								</select>

								{#if selectedModelIdx === 0}
									<button
										class="  self-center {selectedModelIdx === 0
											? 'mr-3'
											: 'mr-7'} disabled:text-gray-600 disabled:hover:text-gray-600"
										disabled={selectedModels.length === 3 || messages.length != 0}
										on:click={() => {
											if (selectedModels.length < 3) {
												selectedModels = [...selectedModels, ''];
											}
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
											<path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m6-6H6" />
										</svg>
									</button>
								{:else}
									<button
										class="  self-center disabled:text-gray-600 disabled:hover:text-gray-600 {selectedModelIdx ===
										0
											? 'mr-3'
											: 'mr-7'}"
										disabled={messages.length != 0}
										on:click={() => {
											selectedModels.splice(selectedModelIdx, 1);
											selectedModels = selectedModels;
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
											<path stroke-linecap="round" stroke-linejoin="round" d="M19.5 12h-15" />
										</svg>
									</button>
								{/if}

								{#if selectedModelIdx === 0}
									<button
										class=" self-center dark:hover:text-gray-300"
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
								{/if}
							</div>
						{/each}
					</div>

					<div class="text-left mt-1.5 text-xs text-gray-500">
						<button on:click={saveDefaultModel}> Set as default</button>
					</div>
				</div>

				<div class=" h-full mt-10 mb-32 w-full flex flex-col">
					{#if messages.length == 0}
						<div class="m-auto text-center max-w-md pb-56 px-2">
							<div class="flex justify-center mt-8">
								<img src="/ollama.png" class=" w-16 invert-[10%] dark:invert-[100%] rounded-full" />
							</div>
							<div class=" mt-1 text-2xl text-gray-800 dark:text-gray-100 font-semibold">
								How can I help you today?
							</div>
						</div>
					{:else}
						{#each messages as message, messageIdx}
							<div class=" w-full">
								<div class="flex justify-between px-5 mb-3 max-w-3xl mx-auto rounded-lg group">
									<div class=" flex w-full">
										<div class=" mr-4">
											<img
												src="{message.role == 'user'
													? settings.gravatarUrl
														? settings.gravatarUrl
														: '/user'
													: '/favicon'}.png"
												class=" max-w-[28px] object-cover rounded-full"
											/>
										</div>

										<div class="w-full">
											<div class=" self-center font-bold mb-0.5">
												{#if message.role === 'user'}
													You
												{:else}
													Ollama <span class=" text-gray-500 text-sm font-medium"
														>{message.model ? ` ${message.model}` : ''}</span
													>
												{/if}
											</div>

											{#if message.role !== 'user' && message.content === ''}
												<div class="w-full mt-3">
													<div class="animate-pulse flex w-full">
														<div class="space-y-2 w-full">
															<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded mr-14" />

															<div class="grid grid-cols-3 gap-4">
																<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-2" />
																<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1" />
															</div>
															<div class="grid grid-cols-4 gap-4">
																<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1" />
																<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-2" />
																<div
																	class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1 mr-4"
																/>
															</div>

															<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded" />
														</div>
													</div>
												</div>
											{:else}
												<div
													class="prose chat-{message.role} w-full max-w-full dark:prose-invert prose-headings:my-0 prose-p:my-0 prose-p:-mb-4 prose-pre:my-0 prose-table:my-0 prose-blockquote:my-0 prose-img:my-0 prose-ul:-my-4 prose-ol:-my-4 prose-li:-my-3 prose-ul:-mb-6 prose-ol:-mb-6 prose-li:-mb-4 whitespace-pre-line"
												>
													{#if message.role == 'user'}
														{#if message?.edit === true}
															<div class=" w-full">
																<textarea
																	class=" bg-transparent outline-none w-full resize-none"
																	bind:value={history.messages[message.id].editedContent}
																	on:input={(e) => {
																		e.target.style.height = '';
																		e.target.style.height = `${e.target.scrollHeight}px`;
																	}}
																	on:focus={(e) => {
																		e.target.style.height = '';
																		e.target.style.height = `${e.target.scrollHeight}px`;
																	}}
																/>

																<div class=" flex justify-end space-x-2 text-sm font-medium">
																	<button
																		class="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded-lg"
																		on:click={() => {
																			confirmEditMessage(message.id);
																		}}
																	>
																		Save & Submit
																	</button>

																	<button
																		class=" px-4 py-2.5 hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-100 transition outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg"
																		on:click={() => {
																			cancelEditMessage(message.id);
																		}}
																	>
																		Cancel
																	</button>
																</div>
															</div>
														{:else}
															<div class="w-full">
																{message.content}

																<div class=" flex justify-start space-x-1">
																	{#if message.parentId !== null && message.parentId in history.messages && (history.messages[message.parentId]?.childrenIds.length ?? 0) > 1}
																		<div class="flex self-center">
																			<button
																				class="self-center"
																				on:click={() => {
																					showPreviousMessage(message);
																				}}
																			>
																				<svg
																					xmlns="http://www.w3.org/2000/svg"
																					viewBox="0 0 20 20"
																					fill="currentColor"
																					class="w-4 h-4"
																				>
																					<path
																						fill-rule="evenodd"
																						d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z"
																						clip-rule="evenodd"
																					/>
																				</svg>
																			</button>

																			<div class="text-xs font-bold self-center">
																				{history.messages[message.parentId].childrenIds.indexOf(
																					message.id
																				) + 1} / {history.messages[message.parentId].childrenIds
																					.length}
																			</div>

																			<button
																				class="self-center"
																				on:click={() => {
																					showNextMessage(message);
																				}}
																			>
																				<svg
																					xmlns="http://www.w3.org/2000/svg"
																					viewBox="0 0 20 20"
																					fill="currentColor"
																					class="w-4 h-4"
																				>
																					<path
																						fill-rule="evenodd"
																						d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
																						clip-rule="evenodd"
																					/>
																				</svg>
																			</button>
																		</div>
																	{:else if message.parentId === null && Object.values(history.messages).filter((message) => message.parentId === null).length > 1}
																		<div class="flex self-center">
																			<button
																				class="self-center"
																				on:click={() => {
																					showPreviousMessage(message);
																				}}
																			>
																				<svg
																					xmlns="http://www.w3.org/2000/svg"
																					viewBox="0 0 20 20"
																					fill="currentColor"
																					class="w-4 h-4"
																				>
																					<path
																						fill-rule="evenodd"
																						d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z"
																						clip-rule="evenodd"
																					/>
																				</svg>
																			</button>

																			<div class="text-xs font-bold self-center">
																				{Object.values(history.messages)
																					.filter((message) => message.parentId === null)
																					.map((message) => message.id)
																					.indexOf(message.id) + 1} / {Object.values(
																					history.messages
																				).filter((message) => message.parentId === null).length}
																			</div>

																			<button
																				class="self-center"
																				on:click={() => {
																					showNextMessage(message);
																				}}
																			>
																				<svg
																					xmlns="http://www.w3.org/2000/svg"
																					viewBox="0 0 20 20"
																					fill="currentColor"
																					class="w-4 h-4"
																				>
																					<path
																						fill-rule="evenodd"
																						d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
																						clip-rule="evenodd"
																					/>
																				</svg>
																			</button>
																		</div>
																	{/if}

																	<button
																		class="invisible group-hover:visible p-1 rounded dark:hover:bg-gray-800 transition"
																		on:click={() => {
																			editMessageHandler(message.id);
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
																				d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487zm0 0L19.5 7.125"
																			/>
																		</svg>
																	</button>
																</div>
															</div>
														{/if}
													{:else}
														<div class="w-full">
															{@html marked(message.content.replace('\\\\', '\\\\\\'))}

															{#if message.done}
																<div class=" flex justify-start space-x-1 -mt-2">
																	{#if message.parentId !== null && message.parentId in history.messages && (history.messages[message.parentId]?.childrenIds.length ?? 0) > 1}
																		<div class="flex self-center">
																			<button
																				class="self-center"
																				on:click={() => {
																					showPreviousMessage(message);
																				}}
																			>
																				<svg
																					xmlns="http://www.w3.org/2000/svg"
																					viewBox="0 0 20 20"
																					fill="currentColor"
																					class="w-4 h-4"
																				>
																					<path
																						fill-rule="evenodd"
																						d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z"
																						clip-rule="evenodd"
																					/>
																				</svg>
																			</button>

																			<div class="text-xs font-bold self-center">
																				{history.messages[message.parentId].childrenIds.indexOf(
																					message.id
																				) + 1} / {history.messages[message.parentId].childrenIds
																					.length}
																			</div>

																			<button
																				class="self-center"
																				on:click={() => {
																					showNextMessage(message);
																				}}
																			>
																				<svg
																					xmlns="http://www.w3.org/2000/svg"
																					viewBox="0 0 20 20"
																					fill="currentColor"
																					class="w-4 h-4"
																				>
																					<path
																						fill-rule="evenodd"
																						d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
																						clip-rule="evenodd"
																					/>
																				</svg>
																			</button>
																		</div>
																	{/if}
																	<button
																		class="{messageIdx + 1 === messages.length
																			? 'visible'
																			: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
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

																	<button
																		class="{messageIdx + 1 === messages.length
																			? 'visible'
																			: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
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
																		class="{messageIdx + 1 === messages.length
																			? 'visible'
																			: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
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

																	{#if messageIdx + 1 === messages.length}
																		<button
																			class="{messageIdx + 1 === messages.length
																				? 'visible'
																				: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
																			on:click={regenerateResponse}
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
																					d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
																				/>
																			</svg>
																		</button>
																	{/if}
																</div>
															{/if}
														</div>
													{/if}
												</div>
											{/if}
										</div>
										<!-- {} -->
									</div>
								</div>
							</div>
						{/each}
					{/if}
				</div>
			</div>

			<div class="fixed bottom-0 w-full">
				<div class="  pt-5">
					<div class="max-w-3xl px-2.5 pt-2.5 -mb-0.5 mx-auto inset-x-0">
						{#if messages.length == 0 && suggestions !== 'false'}
							<Suggestions {submitPrompt} />
						{/if}

						{#if autoScroll === false && messages.length > 0}
							<div class=" flex justify-center mb-4">
								<button
									class=" bg-white/20 p-1.5 rounded-full"
									on:click={() => {
										window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
									}}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 20 20"
										fill="currentColor"
										class="w-5 h-5"
									>
										<path
											fill-rule="evenodd"
											d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							</div>
						{/if}

						<div class="bg-gradient-to-t from-white dark:from-gray-800 from-40% pb-2">
							<form
								class=" flex relative w-full"
								on:submit|preventDefault={() => {
									submitPrompt(prompt);
								}}
							>
								<textarea
									id="chat-textarea"
									class="rounded-xl dark:bg-gray-800 dark:text-gray-100 outline-none border dark:border-gray-600 w-full py-3
									{fileUploadEnabled ? 'pl-12' : 'pl-5'} {speechRecognitionEnabled ? 'pr-20' : 'pr-12'} resize-none"
									placeholder={speechRecognitionListening ? 'Listening...' : 'Send a message'}
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
									on:input={(e) => {
										e.target.style.height = '';
										e.target.style.height = Math.min(e.target.scrollHeight, 200) + 2 + 'px';
									}}
								/>

								{#if fileUploadEnabled}
									<div class=" absolute left-0 bottom-0">
										<div class="pl-2.5 pb-[9px]">
											<button
												class="  text-gray-600 dark:text-gray-200 transition rounded-lg p-1.5"
												type="button"
												on:click={() => {
													console.log('file');
												}}
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													viewBox="0 0 20 20"
													fill="currentColor"
													class="w-5 h-5"
												>
													<path
														fill-rule="evenodd"
														d="M15.621 4.379a3 3 0 00-4.242 0l-7 7a3 3 0 004.241 4.243h.001l.497-.5a.75.75 0 011.064 1.057l-.498.501-.002.002a4.5 4.5 0 01-6.364-6.364l7-7a4.5 4.5 0 016.368 6.36l-3.455 3.553A2.625 2.625 0 119.52 9.52l3.45-3.451a.75.75 0 111.061 1.06l-3.45 3.451a1.125 1.125 0 001.587 1.595l3.454-3.553a3 3 0 000-4.242z"
														clip-rule="evenodd"
													/>
												</svg>
											</button>
										</div>
									</div>
								{/if}

								<div class=" absolute right-0 bottom-0">
									<div class="pr-2.5 pb-[9px]">
										{#if messages.length == 0 || messages.at(-1).done == true}
											{#if speechRecognitionEnabled}
												<button
													class=" text-gray-600 dark:text-gray-300 transition rounded-lg p-1 mr-0.5"
													type="button"
													on:click={() => {
														speechRecognitionHandler();
													}}
												>
													{#if speechRecognitionListening}
														<svg
															class=" w-5 h-5 translate-y-[0.5px]"
															fill="currentColor"
															viewBox="0 0 24 24"
															xmlns="http://www.w3.org/2000/svg"
															><style>
																.spinner_qM83 {
																	animation: spinner_8HQG 1.05s infinite;
																}
																.spinner_oXPr {
																	animation-delay: 0.1s;
																}
																.spinner_ZTLf {
																	animation-delay: 0.2s;
																}
																@keyframes spinner_8HQG {
																	0%,
																	57.14% {
																		animation-timing-function: cubic-bezier(0.33, 0.66, 0.66, 1);
																		transform: translate(0);
																	}
																	28.57% {
																		animation-timing-function: cubic-bezier(0.33, 0, 0.66, 0.33);
																		transform: translateY(-6px);
																	}
																	100% {
																		transform: translate(0);
																	}
																}
															</style><circle class="spinner_qM83" cx="4" cy="12" r="2.5" /><circle
																class="spinner_qM83 spinner_oXPr"
																cx="12"
																cy="12"
																r="2.5"
															/><circle
																class="spinner_qM83 spinner_ZTLf"
																cx="20"
																cy="12"
																r="2.5"
															/></svg
														>
													{:else}
														<svg
															xmlns="http://www.w3.org/2000/svg"
															viewBox="0 0 20 20"
															fill="currentColor"
															class="w-5 h-5 translate-y-[0.5px]"
														>
															<path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
															<path
																d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z"
															/>
														</svg>
													{/if}
												</button>
											{/if}
											<button
												class="{prompt !== ''
													? 'bg-black text-white hover:bg-gray-900 dark:bg-white dark:text-black dark:hover:bg-gray-100 '
													: 'text-white bg-gray-100 dark:text-gray-800 dark:bg-gray-600 disabled'} transition rounded-lg p-1"
												type="submit"
												disabled={prompt === ''}
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													viewBox="0 0 20 20"
													fill="currentColor"
													class="w-5 h-5"
												>
													<path
														fill-rule="evenodd"
														d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
														clip-rule="evenodd"
													/>
												</svg>
											</button>
										{:else}
											<button
												class="bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-800 transition rounded-lg p-1.5"
												on:click={stopResponse}
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													viewBox="0 0 24 24"
													fill="currentColor"
													class="w-5 h-5"
												>
													<path
														fill-rule="evenodd"
														d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm6-2.438c0-.724.588-1.312 1.313-1.312h4.874c.725 0 1.313.588 1.313 1.313v4.874c0 .725-.588 1.313-1.313 1.313H9.564a1.312 1.312 0 01-1.313-1.313V9.564z"
														clip-rule="evenodd"
													/>
												</svg>
											</button>
										{/if}
									</div>
								</div>
							</form>

							<div class="mt-1.5 text-xs text-gray-500 text-center">
								LLMs can make mistakes. Verify important information.
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
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
