<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';

	import { chats, config, modelfiles, settings, user } from '$lib/stores';
	import { tick } from 'svelte';

	import toast from 'svelte-french-toast';
	import { getChatList, updateChatById } from '$lib/apis/chats';

	import UserMessage from './Messages/UserMessage.svelte';
	import ResponseMessage from './Messages/ResponseMessage.svelte';
	import Placeholder from './Messages/Placeholder.svelte';
	import Spinner from '../common/Spinner.svelte';

	export let chatId = '';
	export let sendPrompt: Function;
	export let regenerateResponse: Function;

	export let processing = '';
	export let bottomPadding = false;
	export let autoScroll;
	export let selectedModels;
	export let history = {};
	export let messages = [];

	export let selectedModelfiles = [];

	$: if (autoScroll && bottomPadding) {
		(async () => {
			await tick();
			window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
		})();
	}

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

	const confirmEditMessage = async (messageId, content) => {
		let userPrompt = content;
		let userMessageId = uuidv4();

		let userMessage = {
			id: userMessageId,
			parentId: history.messages[messageId].parentId,
			childrenIds: [],
			role: 'user',
			content: userPrompt,
			...(history.messages[messageId].files && { files: history.messages[messageId].files })
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
		await sendPrompt(userPrompt, userMessageId, chatId);
	};

	const confirmEditResponseMessage = async (messageId, content) => {
		history.messages[messageId].originalContent = history.messages[messageId].content;
		history.messages[messageId].content = content;

		await tick();

		await updateChatById(localStorage.token, chatId, {
			messages: messages,
			history: history
		});

		await chats.set(await getChatList(localStorage.token));
	};

	const rateMessage = async (messageId, rating) => {
		history.messages[messageId].rating = rating;
		await tick();
		await updateChatById(localStorage.token, chatId, {
			messages: messages,
			history: history
		});

		await chats.set(await getChatList(localStorage.token));
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
	<Placeholder models={selectedModels} modelfiles={selectedModelfiles} />
{:else}
	{#key chatId}
		{#each messages as message, messageIdx}
			<div class=" w-full">
				<div class="flex flex-col justify-between px-5 mb-3 max-w-3xl mx-auto rounded-lg group">
					{#if message.role === 'user'}
						<UserMessage
							user={$user}
							{message}
							siblings={message.parentId !== null
								? history.messages[message.parentId]?.childrenIds ?? []
								: Object.values(history.messages)
										.filter((message) => message.parentId === null)
										.map((message) => message.id) ?? []}
							{confirmEditMessage}
							{showPreviousMessage}
							{showNextMessage}
							{copyToClipboard}
						/>

						{#if messages.length - 1 === messageIdx && processing !== ''}
							<div class="flex my-2.5 ml-12 items-center w-fit space-x-2.5">
								<div class=" dark:text-blue-100">
									<svg
										class=" w-4 h-4 translate-y-[0.5px]"
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
										/><circle class="spinner_qM83 spinner_ZTLf" cx="20" cy="12" r="2.5" /></svg
									>
								</div>
								<div class=" text-sm font-medium">
									{processing}
								</div>
							</div>
						{/if}
					{:else}
						<ResponseMessage
							{message}
							modelfiles={selectedModelfiles}
							siblings={history.messages[message.parentId]?.childrenIds ?? []}
							isLastMessage={messageIdx + 1 === messages.length}
							{confirmEditResponseMessage}
							{showPreviousMessage}
							{showNextMessage}
							{rateMessage}
							{copyToClipboard}
							{regenerateResponse}
						/>
					{/if}
				</div>
			</div>
		{/each}

		{#if bottomPadding}
			<div class=" mb-10" />
		{/if}
	{/key}
{/if}
