<script lang="ts">
	import { marked } from 'marked';

	import tippy from 'tippy.js';
	import hljs from 'highlight.js';
	import 'highlight.js/styles/github-dark.min.css';
	import auto_render from 'katex/dist/contrib/auto-render.mjs';
	import 'katex/dist/katex.min.css';

	import Name from './Name.svelte';
	import ProfileImage from './ProfileImage.svelte';
	import Skeleton from './Skeleton.svelte';
	import { onMount, tick } from 'svelte';

	export let modelfiles = [];
	export let message;
	export let siblings;

	export let isLastMessage = true;

	export let confirmEditResponseMessage: Function;
	export let showPreviousMessage: Function;
	export let showNextMessage: Function;
	export let rateMessage: Function;

	export let copyToClipboard: Function;
	export let regenerateResponse: Function;

	let edit = false;
	let editedContent = '';

	let tooltipInstance = null;
	let speaking = null;

	$: if (message) {
		renderStyling();
	}

	const renderStyling = async () => {
		await tick();

		if (tooltipInstance) {
			tooltipInstance[0].destroy();
		}

		renderLatex();
		hljs.highlightAll();
		createCopyCodeBlockButton();

		if (message.info) {
			tooltipInstance = tippy(`#info-${message.id}`, {
				content: `<span class="text-xs" id="tooltip-${message.id}">token/s: ${
					`${
						Math.round(
							((message.info.eval_count ?? 0) / (message.info.eval_duration / 1000000000)) * 100
						) / 100
					} tokens` ?? 'N/A'
				}<br/>
                    total_duration: ${
											Math.round(((message.info.total_duration ?? 0) / 1000000) * 100) / 100 ??
											'N/A'
										}ms<br/>
                    load_duration: ${
											Math.round(((message.info.load_duration ?? 0) / 1000000) * 100) / 100 ?? 'N/A'
										}ms<br/>
                    prompt_eval_count: ${message.info.prompt_eval_count ?? 'N/A'}<br/>
                    prompt_eval_duration: ${
											Math.round(((message.info.prompt_eval_duration ?? 0) / 1000000) * 100) /
												100 ?? 'N/A'
										}ms<br/>
                    eval_count: ${message.info.eval_count ?? 'N/A'}<br/>
                    eval_duration: ${
											Math.round(((message.info.eval_duration ?? 0) / 1000000) * 100) / 100 ?? 'N/A'
										}ms</span>`,
				allowHTML: true
			});
		}
	};

	const createCopyCodeBlockButton = () => {
		// use a class selector if available
		let blocks = document.querySelectorAll('pre');

		blocks.forEach((block) => {
			// only add button if browser supports Clipboard API

			if (block.childNodes.length < 2 && block.id !== 'user-message') {
				let code = block.querySelector('code');
				code.style.borderTopRightRadius = 0;
				code.style.borderTopLeftRadius = 0;
				code.style.whiteSpace = 'pre';

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
				button.className = 'copy-code-button';
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
			}
		});

		async function copyCode(block, button) {
			let code = block.querySelector('code');
			let text = code.innerText;

			await copyToClipboard(text);

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

	const toggleSpeakMessage = async () => {
		if (speaking) {
			speechSynthesis.cancel();
			speaking = null;
		} else {
			speaking = true;
			const speak = new SpeechSynthesisUtterance(message.content);
			speechSynthesis.speak(speak);
		}
	};

	const editMessageHandler = async () => {
		edit = true;
		editedContent = message.content;

		await tick();
		const editElement = document.getElementById(`message-edit-${message.id}`);

		editElement.style.height = '';
		editElement.style.height = `${editElement.scrollHeight}px`;
	};

	const editMessageConfirmHandler = async () => {
		confirmEditResponseMessage(message.id, editedContent);

		edit = false;
		editedContent = '';

		await tick();
		renderStyling();
	};

	const cancelEditMessage = async () => {
		edit = false;
		editedContent = '';
		await tick();
		renderStyling();
	};

	onMount(async () => {
		await tick();
		renderStyling();
	});
</script>

<div class=" flex w-full message-{message.id}">
	<ProfileImage src={modelfiles[message.model]?.imageUrl ?? '/favicon.png'} />

	<div class="w-full overflow-hidden">
		<Name>
			{#if message.model in modelfiles}
				{modelfiles[message.model]?.title}
			{:else}
				Ollama <span class=" text-gray-500 text-sm font-medium"
					>{message.model ? ` ${message.model}` : ''}</span
				>
			{/if}
		</Name>

		{#if message.content === ''}
			<Skeleton />
		{:else}
			<div
				class="prose chat-{message.role} w-full max-w-full dark:prose-invert prose-headings:my-0 prose-p:my-0 prose-p:-mb-4 prose-pre:my-0 prose-table:my-0 prose-blockquote:my-0 prose-img:my-0 prose-ul:-my-4 prose-ol:-my-4 prose-li:-my-3 prose-ul:-mb-6 prose-ol:-mb-6 prose-li:-mb-4 whitespace-pre-line"
			>
				<div>
					{#if edit === true}
						<div class=" w-full">
							<textarea
								id="message-edit-{message.id}"
								class=" bg-transparent outline-none w-full resize-none"
								bind:value={editedContent}
								on:input={(e) => {
									e.target.style.height = `${e.target.scrollHeight}px`;
								}}
							/>

							<div class=" mt-2 mb-1 flex justify-center space-x-2 text-sm font-medium">
								<button
									class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded-lg"
									on:click={() => {
										editMessageConfirmHandler();
									}}
								>
									Save
								</button>

								<button
									class=" px-4 py-2 hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-100 transition outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg"
									on:click={() => {
										cancelEditMessage();
									}}
								>
									Cancel
								</button>
							</div>
						</div>
					{:else}
						<div class="w-full">
							{#if message?.error === true}
								<div
									class="flex mt-2 mb-4 space-x-2 border px-4 py-3 border-red-800 bg-red-800/30 font-medium rounded-lg"
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="w-5 h-5 self-center"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
										/>
									</svg>

									<div class=" self-center">
										{message.content}
									</div>
								</div>
							{:else}
								{@html marked(message.content.replaceAll('\\', '\\\\'))}
							{/if}

							{#if message.done}
								<div class=" flex justify-start space-x-1 -mt-2">
									{#if siblings.length > 1}
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
												{siblings.indexOf(message.id) + 1} / {siblings.length}
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
										class="{isLastMessage
											? 'visible'
											: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
										on:click={() => {
											editMessageHandler();
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
										class="{isLastMessage
											? 'visible'
											: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition copy-response-button"
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
										class="{isLastMessage
											? 'visible'
											: 'invisible group-hover:visible'} p-1 rounded {message.rating === 1
											? 'bg-gray-100 dark:bg-gray-900'
											: ''} transition"
										on:click={() => {
											rateMessage(message.id, 1);
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
										class="{isLastMessage
											? 'visible'
											: 'invisible group-hover:visible'} p-1 rounded {message.rating === -1
											? 'bg-gray-100 dark:bg-gray-900'
											: ''} transition"
										on:click={() => {
											rateMessage(message.id, -1);
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
										class="{isLastMessage
											? 'visible'
											: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition"
										on:click={() => {
											toggleSpeakMessage(message);
										}}
									>
										{#if speaking}
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
													d="M17.25 9.75 19.5 12m0 0 2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6 4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z"
												/>
											</svg>
										{:else}
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
										{/if}
									</button>

									{#if message.info}
										<button
											class=" {isLastMessage
												? 'visible'
												: 'invisible group-hover:visible'} p-1 rounded dark:hover:bg-gray-800 transition whitespace-pre-wrap"
											on:click={() => {
												console.log(message);
											}}
											id="info-{message.id}"
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
													d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"
												/>
											</svg>
										</button>
									{/if}

									{#if isLastMessage}
										<button
											type="button"
											class="{isLastMessage
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
			</div>
		{/if}
	</div>
</div>
