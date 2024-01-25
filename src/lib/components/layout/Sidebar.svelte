<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';

	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { goto, invalidateAll } from '$app/navigation';
	import { page } from '$app/stores';
	import { user, chats, settings, showSettings, chatId, tags } from '$lib/stores';
	import { onMount } from 'svelte';
	import {
		deleteChatById,
		getChatList,
		getChatListByTagName,
		updateChatById
	} from '$lib/apis/chats';

	let show = false;
	let navElement;

	let title: string = 'Ollama Web UI';
	let search = '';

	let chatDeleteId = null;
	let chatTitleEditId = null;
	let chatTitle = '';

	let showDropdown = false;

	onMount(async () => {
		if (window.innerWidth > 1280) {
			show = true;
		}

		await chats.set(await getChatList(localStorage.token));

		tags.subscribe(async (value) => {
			if (value.length === 0) {
				await chats.set(await getChatList(localStorage.token));
			}
		});
	});

	const loadChat = async (id) => {
		goto(`/c/${id}`);
	};

	const editChatTitle = async (id, _title) => {
		title = _title;

		await updateChatById(localStorage.token, id, {
			title: _title
		});
		await chats.set(await getChatList(localStorage.token));
	};

	const deleteChat = async (id) => {
		goto('/');

		await deleteChatById(localStorage.token, id);
		await chats.set(await getChatList(localStorage.token));
	};

	const saveSettings = async (updated) => {
		await settings.set({ ...$settings, ...updated });
		localStorage.setItem('settings', JSON.stringify($settings));
		location.href = '/';
	};
</script>

<div
	bind:this={navElement}
	class="h-screen {show
		? ''
		: '-translate-x-[260px]'}  w-[260px] fixed top-0 left-0 z-40 transition bg-black text-gray-200 shadow-2xl text-sm
        "
>
	<div class="py-2.5 my-auto flex flex-col justify-between h-screen">
		<div class="px-2.5 flex justify-center space-x-2">
			<button
				id="sidebar-new-chat-button"
				class="flex-grow flex justify-between rounded-md px-3 py-2 mt-1 hover:bg-gray-900 transition"
				on:click={async () => {
					goto('/');

					const newChatButton = document.getElementById('new-chat-button');

					if (newChatButton) {
						newChatButton.click();
					}
				}}
			>
				<div class="flex self-center">
					<div class="self-center mr-3.5">
						<img src="/ollama.png" class=" w-5 invert-[100%] rounded-full" />
					</div>

					<div class=" self-center font-medium text-sm">New Chat</div>
				</div>

				<div class="self-center">
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
				</div>
			</button>
		</div>

		{#if $user?.role === 'admin'}
			<div class="px-2.5 flex justify-center mt-0.5">
				<button
					class="flex-grow flex space-x-3 rounded-md px-3 py-2 hover:bg-gray-900 transition"
					on:click={async () => {
						goto('/modelfiles');
					}}
				>
					<div class="self-center">
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
								d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 0 0 2.25-2.25V6a2.25 2.25 0 0 0-2.25-2.25H6A2.25 2.25 0 0 0 3.75 6v2.25A2.25 2.25 0 0 0 6 10.5Zm0 9.75h2.25A2.25 2.25 0 0 0 10.5 18v-2.25a2.25 2.25 0 0 0-2.25-2.25H6a2.25 2.25 0 0 0-2.25 2.25V18A2.25 2.25 0 0 0 6 20.25Zm9.75-9.75H18a2.25 2.25 0 0 0 2.25-2.25V6A2.25 2.25 0 0 0 18 3.75h-2.25A2.25 2.25 0 0 0 13.5 6v2.25a2.25 2.25 0 0 0 2.25 2.25Z"
							/>
						</svg>
					</div>

					<div class="flex self-center">
						<div class=" self-center font-medium text-sm">Modelfiles</div>
					</div>
				</button>
			</div>

			<div class="px-2.5 flex justify-center">
				<button
					class="flex-grow flex space-x-3 rounded-md px-3 py-2 hover:bg-gray-900 transition"
					on:click={async () => {
						goto('/prompts');
					}}
				>
					<div class="self-center">
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
								d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125"
							/>
						</svg>
					</div>

					<div class="flex self-center">
						<div class=" self-center font-medium text-sm">Prompts</div>
					</div>
				</button>
			</div>

			<div class="px-2.5 flex justify-center mb-1">
				<button
					class="flex-grow flex space-x-3 rounded-md px-3 py-2 hover:bg-gray-900 transition"
					on:click={async () => {
						goto('/documents');
					}}
				>
					<div class="self-center">
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
								d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75"
							/>
						</svg>
					</div>

					<div class="flex self-center">
						<div class=" self-center font-medium text-sm">Documents</div>
					</div>
				</button>
			</div>
		{/if}

		<div class="relative flex flex-col flex-1 overflow-y-auto">
			{#if !($settings.saveChatHistory ?? true)}
				<div class="absolute z-40 w-full h-full bg-black/90 flex justify-center">
					<div class=" text-left px-5 py-2">
						<div class=" font-medium">Chat History is off for this browser.</div>
						<div class="text-xs mt-2">
							When history is turned off, new chats on this browser won't appear in your history on
							any of your devices. <span class=" font-semibold"
								>This setting does not sync across browsers or devices.</span
							>
						</div>

						<div class="mt-3">
							<button
								class="flex justify-center items-center space-x-1.5 px-3 py-2.5 rounded-lg text-xs bg-gray-200 hover:bg-gray-300 transition text-gray-800 font-medium w-full"
								type="button"
								on:click={() => {
									saveSettings({
										saveChatHistory: true
									});
								}}
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 16 16"
									fill="currentColor"
									class="w-3 h-3"
								>
									<path
										fill-rule="evenodd"
										d="M8 1a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-1.5 0v-6.5A.75.75 0 0 1 8 1ZM4.11 3.05a.75.75 0 0 1 0 1.06 5.5 5.5 0 1 0 7.78 0 .75.75 0 0 1 1.06-1.06 7 7 0 1 1-9.9 0 .75.75 0 0 1 1.06 0Z"
										clip-rule="evenodd"
									/>
								</svg>

								<div>Enable Chat History</div>
							</button>
						</div>
					</div>
				</div>
			{/if}

			<div class="px-2.5 mt-1 mb-2 flex justify-center space-x-2">
				<div class="flex w-full" id="chat-search">
					<div class="self-center pl-3 py-2 rounded-l bg-gray-950">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>

					<input
						class="w-full rounded-r py-1.5 pl-2.5 pr-4 text-sm text-gray-300 bg-gray-950 outline-none"
						placeholder="Search"
						bind:value={search}
					/>

					<!-- <div class="self-center pr-3 py-2  bg-gray-900">
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
								d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z"
							/>
						</svg>
					</div> -->
				</div>
			</div>

			{#if $tags.length > 0}
				<div class="px-2.5 mt-0.5 mb-2 flex gap-1 flex-wrap">
					<button
						class="px-2.5 text-xs font-medium bg-gray-900 hover:bg-gray-800 transition rounded-full"
						on:click={async () => {
							await chats.set(await getChatList(localStorage.token));
						}}
					>
						all
					</button>
					{#each $tags as tag}
						<button
							class="px-2.5 text-xs font-medium bg-gray-900 hover:bg-gray-800 transition rounded-full"
							on:click={async () => {
								await chats.set(await getChatListByTagName(localStorage.token, tag.name));
							}}
						>
							{tag.name}
						</button>
					{/each}
				</div>
			{/if}

			<div class="pl-2.5 my-2 flex-1 flex flex-col space-y-1 overflow-y-auto">
				{#each $chats.filter((chat) => {
					if (search === '') {
						return true;
					} else {
						let title = chat.title.toLowerCase();
						const query = search.toLowerCase();

						if (title.includes(query)) {
							return true;
						} else {
							return false;
						}
					}
				}) as chat, i}
					<div class=" w-full pr-2 relative">
						<button
							class=" w-full flex justify-between rounded-md px-3 py-2 hover:bg-gray-900 {chat.id ===
							$chatId
								? 'bg-gray-900'
								: ''} transition whitespace-nowrap text-ellipsis"
							on:click={() => {
								// goto(`/c/${chat.id}`);
								if (chat.id !== chatTitleEditId) {
									chatTitleEditId = null;
									chatTitle = '';
								}

								if (chat.id !== $chatId) {
									loadChat(chat.id);
								}
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
									class=" text-left self-center overflow-hidden {chat.id === $chatId
										? 'w-[120px]'
										: 'w-[180px]'} "
								>
									{#if chatTitleEditId === chat.id}
										<input bind:value={chatTitle} class=" bg-transparent w-full" />
									{:else}
										{chat.title}
									{/if}
								</div>
							</div>
						</button>

						{#if chat.id === $chatId}
							<div class=" absolute right-[22px] top-[10px]">
								{#if chatTitleEditId === chat.id}
									<div class="flex self-center space-x-1.5">
										<button
											class=" self-center hover:text-white transition"
											on:click={() => {
												editChatTitle(chat.id, chatTitle);
												chatTitleEditId = null;
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
												chatTitleEditId = null;
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
								{:else if chatDeleteId === chat.id}
									<div class="flex self-center space-x-1.5">
										<button
											class=" self-center hover:text-white transition"
											on:click={() => {
												deleteChat(chat.id);
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
												chatDeleteId = null;
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
											id="delete-chat-button"
											class=" hidden"
											on:click={() => {
												deleteChat(chat.id);
											}}
										/>
										<button
											class=" self-center hover:text-white transition"
											on:click={() => {
												chatTitle = chat.title;
												chatTitleEditId = chat.id;
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
												chatDeleteId = chat.id;
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
		</div>

		<div class="px-2.5">
			<hr class=" border-gray-900 mb-1 w-full" />

			<div class="flex flex-col">
				{#if $user !== undefined}
					<button
						class=" flex rounded-md py-3 px-3.5 w-full hover:bg-gray-900 transition"
						on:click={() => {
							showDropdown = !showDropdown;
						}}
					>
						<div class=" self-center mr-3">
							<img
								src={$user.profile_image_url}
								class=" max-w-[30px] object-cover rounded-full"
								alt="User profile"
							/>
						</div>
						<div class=" self-center font-semibold">{$user.name}</div>
					</button>

					{#if showDropdown}
						<div
							id="dropdownDots"
							class="absolute z-10 bottom-[70px] 4.5rem rounded-lg shadow w-[240px] bg-gray-900"
						>
							<div class="py-2 w-full">
								{#if $user.role === 'admin'}
									<button
										class="flex py-2.5 px-3.5 w-full hover:bg-gray-800 transition"
										on:click={() => {
											goto('/admin');
											showDropdown = false;
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
													d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
												/>
											</svg>
										</div>
										<div class=" self-center font-medium">Admin Panel</div>
									</button>
								{/if}

								<button
									class="flex py-2.5 px-3.5 w-full hover:bg-gray-800 transition"
									on:click={async () => {
										await showSettings.set(true);
										showDropdown = false;
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

							<hr class=" border-gray-700 m-0 p-0" />

							<div class="py-2 w-full">
								<button
									class="flex py-2.5 px-3.5 w-full hover:bg-gray-800 transition"
									on:click={() => {
										localStorage.removeItem('token');
										location.href = '/auth';
										showDropdown = false;
									}}
								>
									<div class=" self-center mr-3">
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
												d="M6 10a.75.75 0 01.75-.75h9.546l-1.048-.943a.75.75 0 111.004-1.114l2.5 2.25a.75.75 0 010 1.114l-2.5 2.25a.75.75 0 11-1.004-1.114l1.048-.943H6.75A.75.75 0 016 10z"
												clip-rule="evenodd"
											/>
										</svg>
									</div>
									<div class=" self-center font-medium">Sign Out</div>
								</button>
							</div>
						</div>
					{/if}
				{:else}
					<button
						class=" flex rounded-md py-3 px-3.5 w-full hover:bg-gray-900 transition"
						on:click={async () => {
							await showSettings.set(true);
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
				{/if}
			</div>
		</div>
	</div>

	<div
		class="fixed left-0 top-[50dvh] z-40 -translate-y-1/2 transition-transform translate-x-[255px] md:translate-x-[260px] rotate-0"
	>
		<button
			id="sidebar-toggle-button"
			class=" group"
			on:click={() => {
				show = !show;
			}}
			><span class="" data-state="closed"
				><div
					class="flex h-[72px] w-8 items-center justify-center opacity-20 group-hover:opacity-100 transition"
				>
					<div class="flex h-6 w-6 flex-col items-center">
						<div
							class="h-3 w-1 rounded-full bg-[#0f0f0f] dark:bg-white rotate-0 translate-y-[0.15rem] {show
								? 'group-hover:rotate-[15deg]'
								: 'group-hover:rotate-[-15deg]'}"
						/>
						<div
							class="h-3 w-1 rounded-full bg-[#0f0f0f] dark:bg-white rotate-0 translate-y-[-0.15rem] {show
								? 'group-hover:rotate-[-15deg]'
								: 'group-hover:rotate-[15deg]'}"
						/>
					</div>
				</div>
			</span>
		</button>
	</div>
</div>
