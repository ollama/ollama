<script lang="ts">
	import { onMount } from 'svelte';

	let show = false;
	let navElement;
	let importFileInputElement;
	let importFiles;

	export let selectedChatId = '';
	export let title: string = 'Ollama Web UI';
	export let chats = [];

	export let createNewChat: Function;
	export let loadChat: Function;
	export let deleteChat: Function;
	export let editChatTitle: Function;
	export let importChatHistory: Function;
	export let exportChatHistory: Function;
	export let deleteChatHistory: Function;
	export let openSettings: Function;

	let chatTitleEditIdx = null;
	let chatTitle = '';

	let _chats = chats.map((item, idx) => chats[chats.length - 1 - idx]);

	onMount(() => {});

	$: if (chats) {
		_chats = chats.map((item, idx) => chats[chats.length - 1 - idx]);
	}

	$: if (importFiles) {
		console.log(importFiles);

		let reader = new FileReader();
		reader.onload = (event) => {
			let chats = JSON.parse(event.target.result);
			console.log(chats);
			importChatHistory(chats);
		};

		reader.readAsText(importFiles[0]);
	}
</script>

<div
	class=" fixed top-0 flex flex-row justify-center bg-stone-100/5 text-gray-200 backdrop-blur-xl w-full z-30"
>
	<div class="basis-full px-5">
		<nav class="py-3" id="nav">
			<div class="flex flex-row justify-between">
				<div class="pl-2">
					<button
						class=" cursor-pointer p-1 flex hover:bg-gray-700 rounded-lg transition"
						on:click={() => {
							show = !show;
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
									fill-rule="evenodd"
									d="M2 4.75A.75.75 0 012.75 4h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 4.75zM2 10a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 10zm0 5.25a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75a.75.75 0 01-.75-.75z"
									clip-rule="evenodd"
								/>
							</svg>
						</div>
					</button>
				</div>

				<div class=" self-center">
					{title != '' ? title.split(' ').slice(0, 7).join(' ') : 'Ollama Web UI'}
				</div>

				<div class="pr-2">
					<button
						class=" cursor-pointer p-1 flex hover:bg-gray-700 rounded-lg transition"
						on:click={() => {
							createNewChat();
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
									d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z"
								/>
							</svg>
						</div>
					</button>
				</div>
			</div>
		</nav>
	</div>
</div>

<div
	bind:this={navElement}
	class="h-screen {show
		? ''
		: '-translate-x-72'} w-72 fixed top-0 left-0 z-40 transition bg-gray-900 text-gray-200 shadow-2xl text-sm
        "
>
	<div class="py-2.5 my-auto flex flex-col justify-between h-screen">
		<div class="px-2.5 flex justify-center space-x-2">
			<button
				class=" cursor-pointer flex-grow rounded-md border border-gray-600 p-3 flex"
				on:click={() => {
					createNewChat();
				}}
			>
				<div class="self-center mr-2">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 20 20"
						fill="currentColor"
						class="w-5 h-5"
					>
						<path
							d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z"
						/>
					</svg>
				</div>

				<div class=" self-center">New Chat</div>
			</button>

			<button
				class=" cursor-pointer w-12 rounded-md border border-gray-600 flex"
				on:click={() => {
					show = !show;
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
							fill-rule="evenodd"
							d="M3 4.25A2.25 2.25 0 015.25 2h5.5A2.25 2.25 0 0113 4.25v2a.75.75 0 01-1.5 0v-2a.75.75 0 00-.75-.75h-5.5a.75.75 0 00-.75.75v11.5c0 .414.336.75.75.75h5.5a.75.75 0 00.75-.75v-2a.75.75 0 011.5 0v2A2.25 2.25 0 0110.75 18h-5.5A2.25 2.25 0 013 15.75V4.25z"
							clip-rule="evenodd"
						/>
						<path
							fill-rule="evenodd"
							d="M19 10a.75.75 0 00-.75-.75H8.704l1.048-.943a.75.75 0 10-1.004-1.114l-2.5 2.25a.75.75 0 000 1.114l2.5 2.25a.75.75 0 101.004-1.114l-1.048-.943h9.546A.75.75 0 0019 10z"
							clip-rule="evenodd"
						/>
					</svg>
				</div>
			</button>
		</div>

		<div class="pl-2.5 my-3 flex-1 flex flex-col space-y-1 overflow-y-auto">
			{#each _chats as chat, i}
				<div class=" w-full pr-2 relative">
					<button
						class=" w-full flex justify-between rounded-md px-4 py-3 hover:bg-gray-800 {chat.id ===
						selectedChatId
							? 'bg-gray-800'
							: ''} transition whitespace-nowrap text-ellipsis"
						on:click={() => {
							if (chat.id !== chatTitleEditIdx) {
								chatTitleEditIdx = null;
								chatTitle = '';
							}

							loadChat(chat.id);
						}}
					>
						<div class=" flex self-center flex-1">
							<div class=" self-center mr-3">
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
										d="M2.25 12.76c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 011.037-.443 48.282 48.282 0 005.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z"
									/>
								</svg>
							</div>
							<div
								class=" text-left self-center overflow-hidden {chat.id === selectedChatId
									? 'w-[150px]'
									: 'w-[200px]'} "
							>
								{#if chatTitleEditIdx === chat.id}
									<input bind:value={chatTitle} class=" bg-transparent w-full" />
								{:else}
									{chat.title}
								{/if}
							</div>
						</div>
					</button>

					{#if chat.id === selectedChatId}
						<div class=" absolute right-[22px] top-[14px]">
							{#if chatTitleEditIdx === chat.id}
								<div class="flex self-center space-x-1.5">
									<button
										class=" self-center hover:text-white transition"
										on:click={() => {
											editChatTitle(chat.id, chatTitle);
											chatTitleEditIdx = null;
											chatTitle = '';
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
												d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
												clip-rule="evenodd"
											/>
										</svg>
									</button>
									<button
										class=" self-center hover:text-white transition"
										on:click={() => {
											chatTitleEditIdx = null;
											chatTitle = '';
										}}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 20 20"
											fill="currentColor"
											class="w-4 h-4"
										>
											<path
												d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
											/>
										</svg>
									</button>
								</div>
							{:else}
								<div class="flex self-center space-x-1.5">
									<button
										class=" self-center hover:text-white transition"
										on:click={() => {
											chatTitle = chat.title;
											chatTitleEditIdx = chat.id;
											// editChatTitle(chat.id, 'a');
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
										class=" self-center hover:text-white transition"
										on:click={() => {
											deleteChat(chat.id);
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
												d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
											/>
										</svg>
									</button>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/each}
		</div>

		<div class="px-2.5">
			<hr class=" border-gray-800 mb-2 w-full" />

			<div class="flex flex-col">
				<div class="flex">
					<input bind:this={importFileInputElement} bind:files={importFiles} type="file" hidden />
					<button
						class=" flex rounded-md p-3.5 w-full hover:bg-gray-800 transition"
						on:click={() => {
							importFileInputElement.click();
							// importChatHistory();
						}}
					>
						<div class=" self-center mr-3">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="w-5 h-5"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
								/>
							</svg>
						</div>
						<div class=" self-center">Import</div>
					</button>
					<button
						class=" flex rounded-md p-3.5 w-full hover:bg-gray-800 transition"
						on:click={() => {
							exportChatHistory();
						}}
					>
						<div class=" self-center mr-3">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="w-5 h-5"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m.75 12l3 3m0 0l3-3m-3 3v-6m-1.5-9H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
								/>
							</svg>
						</div>
						<div class=" self-center">Export</div>
					</button>
				</div>
				<button
					class=" flex rounded-md p-3.5 w-full hover:bg-gray-800 transition"
					on:click={() => {
						deleteChatHistory();
					}}
				>
					<div class=" self-center mr-3">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="1.5"
							stroke="currentColor"
							class="w-5 h-5"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
							/>
						</svg>
					</div>
					<div class=" self-center">Clear conversations</div>
				</button>
				<button
					class=" flex rounded-md p-3.5 w-full hover:bg-gray-800 transition"
					on:click={() => {
						openSettings();
					}}
				>
					<div class=" self-center mr-3">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="1.5"
							stroke="currentColor"
							class="w-5 h-5"
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
					</div>
					<div class=" self-center font-medium">Settings</div>
				</button>
			</div>
		</div>
	</div>
</div>
