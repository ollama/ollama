<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';

	import { goto } from '$app/navigation';
	import { chatId, db, modelfiles } from '$lib/stores';
	import toast from 'svelte-french-toast';

	export let title: string = 'Ollama Web UI';
	export let shareEnabled: boolean = false;

	const shareChat = async () => {
		const chat = await $db.getChatById($chatId);
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
</script>

<nav
	id="nav"
	class=" fixed py-2.5 top-0 flex flex-row justify-center bg-white/95 dark:bg-gray-800/90 dark:text-gray-200 backdrop-blur-xl w-screen z-30"
>
	<div class=" flex max-w-3xl w-full mx-auto px-3">
		<div class="flex w-full max-w-full">
			<div class="pr-2 self-center">
				<button
					class=" cursor-pointer p-1 flex dark:hover:bg-gray-700 rounded-lg transition"
					on:click={async () => {
						console.log('newChat');
						goto('/');
						await chatId.set(uuidv4());
					}}
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
			<div class=" flex-1 self-center font-medium text-ellipsis whitespace-nowrap overflow-hidden">
				{title != '' ? title : 'Ollama Web UI'}
			</div>

			{#if shareEnabled}
				<div class="pl-2">
					<button
						class=" cursor-pointer p-2 flex dark:hover:bg-gray-700 rounded-lg transition border dark:border-gray-600"
						on:click={async () => {
							shareChat();
						}}
					>
						<div class=" m-auto self-center">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 20 20"
								fill="currentColor"
								class="w-4 h-4"
							>
								<path
									d="M9.25 13.25a.75.75 0 001.5 0V4.636l2.955 3.129a.75.75 0 001.09-1.03l-4.25-4.5a.75.75 0 00-1.09 0l-4.25 4.5a.75.75 0 101.09 1.03L9.25 4.636v8.614z"
								/>
								<path
									d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z"
								/>
							</svg>
						</div>
					</button>
				</div>
			{/if}
		</div>
	</div>
</nav>
