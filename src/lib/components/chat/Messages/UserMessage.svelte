<script lang="ts">
	import { tick } from 'svelte';
	import Name from './Name.svelte';
	import ProfileImage from './ProfileImage.svelte';

	export let user;
	export let message;
	export let siblings;

	export let confirmEditMessage: Function;
	export let showPreviousMessage: Function;
	export let showNextMessage: Function;
	export let copyToClipboard: Function;

	let edit = false;
	let editedContent = '';

	const editMessageHandler = async () => {
		edit = true;
		editedContent = message.content;

		await tick();
		const editElement = document.getElementById(`message-edit-${message.id}`);

		editElement.style.height = '';
		editElement.style.height = `${editElement.scrollHeight}px`;
	};

	const editMessageConfirmHandler = async () => {
		confirmEditMessage(message.id, editedContent);

		edit = false;
		editedContent = '';
	};

	const cancelEditMessage = () => {
		edit = false;
		editedContent = '';
	};
</script>

<div class=" flex w-full">
	<ProfileImage src={user?.profile_image_url ?? '/user.png'} />

	<div class="w-full overflow-hidden">
		<Name>You</Name>

		<div
			class="prose chat-{message.role} w-full max-w-full dark:prose-invert prose-headings:my-0 prose-p:my-0 prose-p:-mb-4 prose-pre:my-0 prose-table:my-0 prose-blockquote:my-0 prose-img:my-0 prose-ul:-my-4 prose-ol:-my-4 prose-li:-my-3 prose-ul:-mb-6 prose-ol:-mb-6 prose-li:-mb-4 whitespace-pre-line"
		>
			{#if message.files}
				<div class="my-3 w-full flex overflow-x-auto space-x-2">
					{#each message.files as file}
						<div>
							{#if file.type === 'image'}
								<img src={file.url} alt="input" class=" max-h-96 rounded-lg" draggable="false" />
							{/if}
						</div>
					{/each}
				</div>
			{/if}

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
							Save & Submit
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
					<pre id="user-message">{message.content}</pre>

					<div class=" flex justify-start space-x-1">
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
							class="invisible group-hover:visible p-1 rounded dark:hover:bg-gray-800 transition"
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
		</div>
	</div>
</div>
