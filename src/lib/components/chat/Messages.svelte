<script lang="ts">
	import { marked } from 'marked';

	import { v4 as uuidv4 } from 'uuid';
	import hljs from 'highlight.js';
	import 'highlight.js/styles/github-dark.min.css';
	import auto_render from 'katex/dist/contrib/auto-render.mjs';
	import 'katex/dist/katex.min.css';

	import { config, db, settings, user } from '$lib/stores';
	import { tick } from 'svelte';

	import toast from 'svelte-french-toast';

	export let sendPrompt: Function;
	export let regenerateResponse: Function;

	export let autoScroll;
	export let history = {};
	export let messages = [];

	$: if (messages && messages.length > 0 && (messages.at(-1).done ?? false)) {
		(async () => {
			await tick();
			renderLatex();
			hljs.highlightAll();
			createCopyCodeBlockButton();
		})();
	}

	const speakMessage = (message) => {
		const speak = new SpeechSynthesisUtterance(message);
		speechSynthesis.speak(speak);
	};

	const createCopyCodeBlockButton = () => {
		// use a class selector if available
		let blocks = document.querySelectorAll('pre');

		blocks.forEach((block) => {
			// only add button if browser supports Clipboard API

			if (navigator.clipboard && block.childNodes.length < 2 && block.id !== 'user-message') {
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
					// { left: '$', right: '$', display: false },
					{ left: '\\(', right: '\\)', display: true },
					{ left: '\\[', right: '\\]', display: true }
				],
				// • rendering keys, e.g.:
				throwOnError: false
			});
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

		$db.updateChatById(chatId, {
			messages: messages,
			history: history
		});
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

		autoScroll = window.innerHeight + window.scrollY >= document.body.offsetHeight - 40;

		setTimeout(() => {
			window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
		}, 100);
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

		autoScroll = window.innerHeight + window.scrollY >= document.body.offsetHeight - 40;
		setTimeout(() => {
			window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
		}, 100);
	};
</script>

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
						{#if message.role === 'user'}
							{#if $config === null || !($config?.auth ?? true)}
								<img
									src="{$settings.gravatarUrl ? $settings.gravatarUrl : '/user'}.png"
									class=" max-w-[28px] object-cover rounded-full"
									alt="User profile"
								/>
							{:else}
								<img
									src={$user ? $user.profile_image_url : '/user.png'}
									class=" max-w-[28px] object-cover rounded-full"
									alt="User profile"
								/>
							{/if}
						{:else}
							<img
								src="/favicon.png"
								class=" max-w-[28px] object-cover rounded-full"
								alt="Ollama profile"
							/>
						{/if}
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
											<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1 mr-4" />
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

											<div class=" mt-2 flex justify-center space-x-2 text-sm font-medium">
												<button
													class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded-lg"
													on:click={() => {
														confirmEditMessage(message.id);
													}}
												>
													Save & Submit
												</button>

												<button
													class=" px-4 py-2 hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-100 transition outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg"
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
											{#if message.files}
												<div class="my-3">
													{#each message.files as file}
														<div>
															{#if file.type === 'image'}
																<img src={file.url} alt="input" class=" max-h-96" />
															{/if}
														</div>
													{/each}
												</div>
											{/if}
											<pre id="user-message">{message.content}</pre>

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
															{history.messages[message.parentId].childrenIds.indexOf(message.id) +
																1} / {history.messages[message.parentId].childrenIds.length}
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
																.indexOf(message.id) + 1} / {Object.values(history.messages).filter(
																(message) => message.parentId === null
															).length}
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

												<button
													class="invisible group-hover:visible p-1 rounded dark:hover:bg-gray-800 transition"
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
															{history.messages[message.parentId].childrenIds.indexOf(message.id) +
																1} / {history.messages[message.parentId].childrenIds.length}
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

												<button
													class="{messageIdx + 1 === messages.length
														? 'visible'
														: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
													on:click={() => {
														speakMessage(message.content);
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
															d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"
														/>
													</svg>
												</button>

												{#if messageIdx + 1 === messages.length}
													<button
														type="button"
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
