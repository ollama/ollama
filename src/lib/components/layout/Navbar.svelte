<script lang="ts">
	import toast from 'svelte-french-toast';
	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { getChatById } from '$lib/apis/chats';
	import { chatId, modelfiles } from '$lib/stores';
	import ShareChatModal from '../chat/ShareChatModal.svelte';
	import TagInput from '../common/Tags/TagInput.svelte';
	import Tags from '../common/Tags.svelte';

	export let initNewChat: Function;
	export let title: string = 'Ollama Web UI';
	export let shareEnabled: boolean = false;

	export let tags = [];
	export let addTag: Function;
	export let deleteTag: Function;

	let showShareChatModal = false;

	let tagName = '';
	let showTagInput = false;

	const shareChat = async () => {
		const chat = (await getChatById(localStorage.token, $chatId)).chat;
		console.log('share', chat);

		toast.success('Redirecting you to OllamaHub');
		const url = 'https://ollamahub.com';
		// const url = 'http://localhost:5173';

		const tab = await window.open(`${url}/chats/upload`, '_blank');
		window.addEventListener(
			'message',
			(event) => {
				if (event.origin !== url) return;
				if (event.data === 'loaded') {
					tab.postMessage(
						JSON.stringify({
							chat: chat,
							modelfiles: $modelfiles.filter((modelfile) => chat.models.includes(modelfile.tagName))
						}),
						'*'
					);
				}
			},
			false
		);
	};

	const downloadChat = async () => {
		const chat = (await getChatById(localStorage.token, $chatId)).chat;
		console.log('download', chat);

		const chatText = chat.messages.reduce((a, message, i, arr) => {
			return `${a}### ${message.role.toUpperCase()}\n${message.content}\n\n`;
		}, '');

		let blob = new Blob([chatText], {
			type: 'text/plain'
		});

		saveAs(blob, `chat-${chat.title}.txt`);
	};
</script>

<ShareChatModal bind:show={showShareChatModal} {downloadChat} {shareChat} />
<nav
	id="nav"
	class=" fixed py-2.5 top-0 flex flex-row justify-center bg-white/95 dark:bg-gray-800/90 dark:text-gray-200 backdrop-blur-xl w-screen z-30"
>
	<div class=" flex max-w-3xl w-full mx-auto px-3">
		<div class="flex items-center w-full max-w-full">
			<div class="pr-2 self-start">
				<button
					id="new-chat-button"
					class=" cursor-pointer p-1.5 flex dark:hover:bg-gray-700 rounded-lg transition"
					on:click={initNewChat}
				>
					<div class=" m-auto self-center">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-5 h-5"
						>
							<path
								d="M5.433 13.917l1.262-3.155A4 4 0 017.58 9.42l6.92-6.918a2.121 2.121 0 013 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 01-.65-.65z"
							/>
							<path
								d="M3.5 5.75c0-.69.56-1.25 1.25-1.25H10A.75.75 0 0010 3H4.75A2.75 2.75 0 002 5.75v9.5A2.75 2.75 0 004.75 18h9.5A2.75 2.75 0 0017 15.25V10a.75.75 0 00-1.5 0v5.25c0 .69-.56 1.25-1.25 1.25h-9.5c-.69 0-1.25-.56-1.25-1.25v-9.5z"
							/>
						</svg>
					</div>
				</button>
			</div>
			<div class=" flex-1 self-center font-medium line-clamp-1">
				<div>
					{title != '' ? title : 'Ollama Web UI'}
				</div>
			</div>

			<div class="pl-2 self-center flex items-center space-x-2">
				{#if shareEnabled}
					<Tags {tags} {deleteTag} {addTag} />

					<button
						class=" cursor-pointer p-1.5 flex dark:hover:bg-gray-700 rounded-lg transition border dark:border-gray-600"
						on:click={async () => {
							showShareChatModal = !showShareChatModal;

							// console.log(showShareChatModal);
						}}
					>
						<div class=" m-auto self-center">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 24 24"
								fill="currentColor"
								class="w-4 h-4"
							>
								<path
									fill-rule="evenodd"
									d="M15.75 4.5a3 3 0 1 1 .825 2.066l-8.421 4.679a3.002 3.002 0 0 1 0 1.51l8.421 4.679a3 3 0 1 1-.729 1.31l-8.421-4.678a3 3 0 1 1 0-4.132l8.421-4.679a3 3 0 0 1-.096-.755Z"
									clip-rule="evenodd"
								/>
							</svg>
						</div>
					</button>
				{/if}
			</div>
		</div>
	</div>
</nav>
