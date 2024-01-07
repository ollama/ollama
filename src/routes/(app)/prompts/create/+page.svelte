<script>
	import toast from 'svelte-french-toast';

	import { goto } from '$app/navigation';
	import { prompts } from '$lib/stores';
	import { onMount, tick } from 'svelte';

	import { createNewPrompt, getPrompts } from '$lib/apis/prompts';

	let loading = false;

	// ///////////
	// Prompt
	// ///////////

	let title = '';
	let command = '';
	let content = '';

	$: command = title !== '' ? `${title.replace(/\s+/g, '-').toLowerCase()}` : '';

	const submitHandler = async () => {
		loading = true;

		if (validateCommandString(command)) {
			const prompt = await createNewPrompt(localStorage.token, command, title, content).catch(
				(error) => {
					toast.error(error);

					return null;
				}
			);

			if (prompt) {
				await prompts.set(await getPrompts(localStorage.token));
				await goto('/prompts');
			}
		} else {
			toast.error('Only alphanumeric characters and hyphens are allowed in the command string.');
		}

		loading = false;
	};

	const validateCommandString = (inputString) => {
		// Regular expression to match only alphanumeric characters and hyphen
		const regex = /^[a-zA-Z0-9-]+$/;

		// Test the input string against the regular expression
		return regex.test(inputString);
	};

	onMount(() => {
		window.addEventListener('message', async (event) => {
			if (
				!['https://ollamahub.com', 'https://www.ollamahub.com', 'http://localhost:5173'].includes(
					event.origin
				)
			)
				return;
			const prompt = JSON.parse(event.data);
			console.log(prompt);

			title = prompt.title;
			await tick();
			content = prompt.content;
			command = prompt.command;
		});

		if (window.opener ?? false) {
			window.opener.postMessage('loaded', '*');
		}
	});
</script>

<div class="min-h-screen w-full flex justify-center dark:text-white">
	<div class=" py-2.5 flex flex-col justify-between w-full">
		<div class="max-w-2xl mx-auto w-full px-3 md:px-0 my-10">
			<div class=" text-2xl font-semibold mb-6">My Prompts</div>

			<button
				class="flex space-x-1"
				on:click={() => {
					history.back();
				}}
			>
				<div class=" self-center">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 20 20"
						fill="currentColor"
						class="w-4 h-4"
					>
						<path
							fill-rule="evenodd"
							d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z"
							clip-rule="evenodd"
						/>
					</svg>
				</div>
				<div class=" self-center font-medium text-sm">Back</div>
			</button>
			<hr class="my-3 dark:border-gray-700" />

			<form
				class="flex flex-col"
				on:submit|preventDefault={() => {
					submitHandler();
				}}
			>
				<div class="my-2">
					<div class=" text-sm font-semibold mb-2">Title*</div>

					<div>
						<input
							class="px-3 py-1.5 text-sm w-full bg-transparent border dark:border-gray-600 outline-none rounded-lg"
							placeholder="Add a short title for this prompt"
							bind:value={title}
							required
						/>
					</div>
				</div>

				<div class="my-2">
					<div class=" text-sm font-semibold mb-2">Command*</div>

					<div class="flex items-center mb-1">
						<div
							class="bg-gray-200 dark:bg-gray-600 font-bold px-3 py-1 border border-r-0 dark:border-gray-600 rounded-l-lg"
						>
							/
						</div>
						<input
							class="px-3 py-1.5 text-sm w-full bg-transparent border dark:border-gray-600 outline-none rounded-r-lg"
							placeholder="short-summary"
							bind:value={command}
							required
						/>
					</div>

					<div class="text-xs text-gray-400 dark:text-gray-500">
						Only <span class=" text-gray-600 dark:text-gray-300 font-medium"
							>alphanumeric characters and hyphens</span
						>
						are allowed; Activate this command by typing "<span
							class=" text-gray-600 dark:text-gray-300 font-medium"
						>
							/{command}
						</span>" to chat input.
					</div>
				</div>

				<div class="my-2">
					<div class="flex w-full justify-between">
						<div class=" self-center text-sm font-semibold">Prompt Content*</div>
					</div>

					<div class="mt-2">
						<div>
							<textarea
								class="px-3 py-1.5 text-sm w-full bg-transparent border dark:border-gray-600 outline-none rounded-lg"
								placeholder={`Write a summary in 50 words that summarizes [topic or keyword].`}
								rows="6"
								bind:value={content}
								required
							/>
						</div>

						<div class="text-xs text-gray-400 dark:text-gray-500">
							Format your variables using square brackets like this: <span
								class=" text-gray-600 dark:text-gray-300 font-medium">[variable]</span
							>
							. Make sure to enclose them with
							<span class=" text-gray-600 dark:text-gray-300 font-medium">'['</span>
							and <span class=" text-gray-600 dark:text-gray-300 font-medium">']'</span> .
						</div>
					</div>
				</div>

				<div class="my-2 flex justify-end">
					<button
						class=" text-sm px-3 py-2 transition rounded-xl {loading
							? ' cursor-not-allowed bg-gray-100 dark:bg-gray-800'
							: ' bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-800'} flex"
						type="submit"
						disabled={loading}
					>
						<div class=" self-center font-medium">Save & Create</div>

						{#if loading}
							<div class="ml-1.5 self-center">
								<svg
									class=" w-4 h-4"
									viewBox="0 0 24 24"
									fill="currentColor"
									xmlns="http://www.w3.org/2000/svg"
									><style>
										.spinner_ajPY {
											transform-origin: center;
											animation: spinner_AtaB 0.75s infinite linear;
										}
										@keyframes spinner_AtaB {
											100% {
												transform: rotate(360deg);
											}
										}
									</style><path
										d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z"
										opacity=".25"
									/><path
										d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z"
										class="spinner_ajPY"
									/></svg
								>
							</div>
						{/if}
					</button>
				</div>
			</form>
		</div>
	</div>
</div>
