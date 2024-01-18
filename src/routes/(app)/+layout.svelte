<script lang="ts">
	import toast from 'svelte-french-toast';
	import { openDB, deleteDB } from 'idb';
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';

	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { getOllamaModels, getOllamaVersion } from '$lib/apis/ollama';
	import { getModelfiles } from '$lib/apis/modelfiles';
	import { getPrompts } from '$lib/apis/prompts';

	import { getOpenAIModels } from '$lib/apis/openai';

	import {
		user,
		showSettings,
		settings,
		models,
		modelfiles,
		prompts,
		documents,
		tags
	} from '$lib/stores';
	import { REQUIRED_OLLAMA_VERSION, WEBUI_API_BASE_URL } from '$lib/constants';

	import SettingsModal from '$lib/components/chat/SettingsModal.svelte';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';
	import { checkVersion } from '$lib/utils';
	import ShortcutsModal from '$lib/components/chat/ShortcutsModal.svelte';
	import { getDocs } from '$lib/apis/documents';
	import { getAllChatTags } from '$lib/apis/chats';

	let ollamaVersion = '';
	let loaded = false;

	let DB = null;
	let localDBChats = [];

	let showShortcuts = false;

	const getModels = async () => {
		let models = [];
		models.push(
			...(await getOllamaModels(localStorage.token).catch((error) => {
				toast.error(error);
				return [];
			}))
		);

		// $settings.OPENAI_API_BASE_URL ?? 'https://api.openai.com/v1',
		// 		$settings.OPENAI_API_KEY

		const openAIModels = await getOpenAIModels(localStorage.token).catch((error) => {
			console.log(error);
			return null;
		});

		models.push(...(openAIModels ? [{ name: 'hr' }, ...openAIModels] : []));

		return models;
	};

	const setOllamaVersion = async (version: string = '') => {
		if (version === '') {
			version = await getOllamaVersion(localStorage.token).catch((error) => {
				return '';
			});
		}

		ollamaVersion = version;

		console.log(ollamaVersion);
		if (checkVersion(REQUIRED_OLLAMA_VERSION, ollamaVersion)) {
			toast.error(`Ollama Version: ${ollamaVersion !== '' ? ollamaVersion : 'Not Detected'}`);
		}
	};

	onMount(async () => {
		if ($user === undefined) {
			await goto('/auth');
		} else if (['user', 'admin'].includes($user.role)) {
			try {
				// Check if IndexedDB exists
				DB = await openDB('Chats', 1);

				if (DB) {
					const chats = await DB.getAllFromIndex('chats', 'timestamp');
					localDBChats = chats.map((item, idx) => chats[chats.length - 1 - idx]);

					if (localDBChats.length === 0) {
						await deleteDB('Chats');
					}

					console.log('localdb', localDBChats);
				}

				console.log(DB);
			} catch (error) {
				// IndexedDB Not Found
				console.log('IDB Not Found');
			}

			console.log();
			await settings.set(JSON.parse(localStorage.getItem('settings') ?? '{}'));

			await modelfiles.set(await getModelfiles(localStorage.token));
			await prompts.set(await getPrompts(localStorage.token));
			await documents.set(await getDocs(localStorage.token));
			await tags.set(await getAllChatTags(localStorage.token));

			modelfiles.subscribe(async () => {
				// should fetch models
				await models.set(await getModels());
			});

			await setOllamaVersion();

			document.addEventListener('keydown', function (event) {
				const isCtrlPressed = event.ctrlKey || event.metaKey; // metaKey is for Cmd key on Mac
				// Check if the Shift key is pressed
				const isShiftPressed = event.shiftKey;

				// Check if Ctrl + Shift + O is pressed
				if (isCtrlPressed && isShiftPressed && event.key.toLowerCase() === 'o') {
					event.preventDefault();
					console.log('newChat');
					document.getElementById('sidebar-new-chat-button')?.click();
				}

				// Check if Shift + Esc is pressed
				if (isShiftPressed && event.key === 'Escape') {
					event.preventDefault();
					console.log('focusInput');
					document.getElementById('chat-textarea')?.focus();
				}

				// Check if Ctrl + Shift + ; is pressed
				if (isCtrlPressed && isShiftPressed && event.key === ';') {
					event.preventDefault();
					console.log('copyLastCodeBlock');
					const button = [...document.getElementsByClassName('copy-code-button')]?.at(-1);
					button?.click();
				}

				// Check if Ctrl + Shift + C is pressed
				if (isCtrlPressed && isShiftPressed && event.key.toLowerCase() === 'c') {
					event.preventDefault();
					console.log('copyLastResponse');
					const button = [...document.getElementsByClassName('copy-response-button')]?.at(-1);
					console.log(button);
					button?.click();
				}

				// Check if Ctrl + Shift + S is pressed
				if (isCtrlPressed && isShiftPressed && event.key.toLowerCase() === 's') {
					event.preventDefault();
					console.log('toggleSidebar');
					document.getElementById('sidebar-toggle-button')?.click();
				}

				// Check if Ctrl + Shift + Backspace is pressed
				if (isCtrlPressed && isShiftPressed && event.key === 'Backspace') {
					event.preventDefault();
					console.log('deleteChat');
					document.getElementById('delete-chat-button')?.click();
				}

				// Check if Ctrl + . is pressed
				if (isCtrlPressed && event.key === '.') {
					event.preventDefault();
					console.log('openSettings');
					document.getElementById('open-settings-button')?.click();
				}

				// Check if Ctrl + / is pressed
				if (isCtrlPressed && event.key === '/') {
					event.preventDefault();
					console.log('showShortcuts');
					document.getElementById('show-shortcuts-button')?.click();
				}
			});

			await tick();
		}

		loaded = true;
	});
</script>

{#if loaded}
	<div class=" hidden lg:flex fixed bottom-0 right-0 px-3 py-3 z-10">
		<button
			id="show-shortcuts-button"
			class="text-gray-600 dark:text-gray-300 bg-gray-300/20 w-6 h-6 flex items-center justify-center text-xs rounded-full"
			on:click={() => {
				showShortcuts = !showShortcuts;
			}}
		>
			?
		</button>
	</div>

	<ShortcutsModal bind:show={showShortcuts} />

	<div class="app relative">
		{#if !['user', 'admin'].includes($user.role)}
			<div class="fixed w-full h-full flex z-50">
				<div
					class="absolute w-full h-full backdrop-blur-md bg-white/20 dark:bg-gray-900/50 flex justify-center"
				>
					<div class="m-auto pb-44 flex flex-col justify-center">
						<div class="max-w-md">
							<div class="text-center dark:text-white text-2xl font-medium z-50">
								Account Activation Pending<br /> Contact Admin for WebUI Access
							</div>

							<div class=" mt-4 text-center text-sm dark:text-gray-200 w-full">
								Your account status is currently pending activation. To access the WebUI, please
								reach out to the administrator. Admins can manage user statuses from the Admin
								Panel.
							</div>

							<div class=" mt-6 mx-auto relative group w-fit">
								<button
									class="relative z-20 flex px-5 py-2 rounded-full bg-white border border-gray-100 dark:border-none hover:bg-gray-100 transition font-medium text-sm"
									on:click={async () => {
										location.href = '/';
									}}
								>
									Check Again
								</button>

								<button
									class="text-xs text-center w-full mt-2 text-gray-400 underline"
									on:click={async () => {
										localStorage.removeItem('token');
										location.href = '/auth';
									}}>Sign Out</button
								>
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if checkVersion(REQUIRED_OLLAMA_VERSION, ollamaVersion ?? '0')}
			<div class="fixed w-full h-full flex z-50">
				<div
					class="absolute w-full h-full backdrop-blur-md bg-white/20 dark:bg-gray-900/50 flex justify-center"
				>
					<div class="m-auto pb-44 flex flex-col justify-center">
						<div class="max-w-md">
							<div class="text-center dark:text-white text-2xl font-medium z-50">
								Connection Issue or Update Needed
							</div>

							<div class=" mt-4 text-center text-sm dark:text-gray-200 w-full">
								Oops! It seems like your Ollama needs a little attention. <br
									class=" hidden sm:flex"
								/>We've detected either a connection hiccup or observed that you're using an older
								version. Ensure you're on the latest Ollama version
								<br class=" hidden sm:flex" />(version
								<span class=" dark:text-white font-medium">{REQUIRED_OLLAMA_VERSION} or higher</span
								>) or check your connection.

								<div class="mt-1 text-sm">
									Trouble accessing Ollama?
									<a
										class=" text-black dark:text-white font-semibold underline"
										href="https://github.com/ollama-webui/ollama-webui#troubleshooting"
										target="_blank"
									>
										Click here for help.
									</a>
								</div>
							</div>

							<div class=" mt-6 mx-auto relative group w-fit">
								<button
									class="relative z-20 flex px-5 py-2 rounded-full bg-white border border-gray-100 dark:border-none hover:bg-gray-100 transition font-medium text-sm"
									on:click={async () => {
										location.href = '/';
										// await setOllamaVersion();
									}}
								>
									Check Again
								</button>

								<button
									class="text-xs text-center w-full mt-2 text-gray-400 underline"
									on:click={async () => {
										await setOllamaVersion(REQUIRED_OLLAMA_VERSION);
									}}>Close</button
								>
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if localDBChats.length > 0}
			<div class="fixed w-full h-full flex z-50">
				<div
					class="absolute w-full h-full backdrop-blur-md bg-white/20 dark:bg-gray-900/50 flex justify-center"
				>
					<div class="m-auto pb-44 flex flex-col justify-center">
						<div class="max-w-md">
							<div class="text-center dark:text-white text-2xl font-medium z-50">
								Important Update<br /> Action Required for Chat Log Storage
							</div>

							<div class=" mt-4 text-center text-sm dark:text-gray-200 w-full">
								Saving chat logs directly to your browser's storage is no longer supported. Please
								take a moment to download and delete your chat logs by clicking the button below.
								Don't worry, you can easily re-import your chat logs to the backend through <span
									class="font-semibold dark:text-white">Settings > Chats > Import Chats</span
								>. This ensures that your valuable conversations are securely saved to your backend
								database. Thank you!
							</div>

							<div class=" mt-6 mx-auto relative group w-fit">
								<button
									class="relative z-20 flex px-5 py-2 rounded-full bg-white border border-gray-100 dark:border-none hover:bg-gray-100 transition font-medium text-sm"
									on:click={async () => {
										let blob = new Blob([JSON.stringify(localDBChats)], {
											type: 'application/json'
										});
										saveAs(blob, `chat-export-${Date.now()}.json`);

										const tx = DB.transaction('chats', 'readwrite');
										await Promise.all([tx.store.clear(), tx.done]);
										await deleteDB('Chats');

										localDBChats = [];
									}}
								>
									Download & Delete
								</button>

								<button
									class="text-xs text-center w-full mt-2 text-gray-400 underline"
									on:click={async () => {
										localDBChats = [];
									}}>Close</button
								>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}

		<div
			class=" text-gray-700 dark:text-gray-100 bg-white dark:bg-gray-800 min-h-screen overflow-auto flex flex-row"
		>
			<Sidebar />
			<SettingsModal bind:show={$showSettings} />
			<slot />
		</div>
	</div>
{/if}

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
