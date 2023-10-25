<script lang="ts">
	import Modal from '../common/Modal.svelte';

	import { API_BASE_URL as BUILD_TIME_API_BASE_URL } from '$lib/constants';
	import toast from 'svelte-french-toast';

	export let show = false;
	export let saveSettings: Function;
	export let getModelTags: Function;

	let API_BASE_URL = BUILD_TIME_API_BASE_URL;
	let system = '';
	let temperature = 0.8;

	let selectedMenu = 'general';
	let modelTag = '';
	let deleteModelTag = '';

	let digest = '';
	let pullProgress = '';

	const splitStream = (splitOn) => {
		let buffer = '';
		return new TransformStream({
			transform(chunk, controller) {
				buffer += chunk;
				const parts = buffer.split(splitOn);
				parts.slice(0, -1).forEach((part) => controller.enqueue(part));
				buffer = parts[parts.length - 1];
			},
			flush(controller) {
				if (buffer) controller.enqueue(buffer);
			}
		});
	};

	const checkOllamaConnection = async () => {
		if (API_BASE_URL === '') {
			API_BASE_URL = BUILD_TIME_API_BASE_URL;
		}
		const res = await getModelTags(API_BASE_URL);

		if (res) {
			toast.success('Server connection verified');
			saveSettings(
				API_BASE_URL,
				system != '' ? system : null,
				temperature != 0.8 ? temperature : null
			);
		}
	};

	const pullModelHandler = async () => {
		const res = await fetch(`${API_BASE_URL}/pull`, {
			method: 'POST',
			headers: {
				'Content-Type': 'text/event-stream'
			},
			body: JSON.stringify({
				name: modelTag
			})
		});

		const reader = res.body
			.pipeThrough(new TextDecoderStream())
			.pipeThrough(splitStream('\n'))
			.getReader();

		while (true) {
			const { value, done } = await reader.read();
			if (done) break;

			try {
				let lines = value.split('\n');

				for (const line of lines) {
					if (line !== '') {
						console.log(line);
						let data = JSON.parse(line);
						console.log(data);

						if (data.error) {
							throw data.error;
						}
						if (data.status) {
							if (!data.status.includes('downloading')) {
								toast.success(data.status);
							} else {
								digest = data.digest;
								if (data.completed) {
									pullProgress = Math.round((data.completed / data.total) * 1000) / 10;
								} else {
									pullProgress = 100;
								}
							}
						}
					}
				}
			} catch (error) {
				console.log(error);
				toast.error(error);
			}
		}

		modelTag = '';
		await getModelTags();
	};

	const deleteModelHandler = async () => {
		const res = await fetch(`${API_BASE_URL}/delete`, {
			method: 'DELETE',
			headers: {
				'Content-Type': 'text/event-stream'
			},
			body: JSON.stringify({
				name: deleteModelTag
			})
		});

		const reader = res.body
			.pipeThrough(new TextDecoderStream())
			.pipeThrough(splitStream('\n'))
			.getReader();

		while (true) {
			const { value, done } = await reader.read();
			if (done) break;

			try {
				let lines = value.split('\n');

				for (const line of lines) {
					if (line !== '' && line !== 'null') {
						console.log(line);
						let data = JSON.parse(line);
						console.log(data);

						if (data.error) {
							throw data.error;
						}
						if (data.status) {
						}
					} else {
						toast.success(`Deleted ${deleteModelTag}`);
					}
				}
			} catch (error) {
				console.log(error);
				toast.error(error);
			}
		}

		deleteModelTag = '';
		await getModelTags();
	};

	$: if (show) {
		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');
		API_BASE_URL = settings.API_BASE_URL ?? BUILD_TIME_API_BASE_URL;
		system = settings.system ?? '';
		temperature = settings.temperature ?? 0.8;
	}
</script>

<Modal bind:show>
	<div class="rounded-lg bg-gray-900">
		<div class=" flex justify-between text-gray-300 px-5 py-4">
			<div class=" text-lg font-medium self-center">Settings</div>
			<button
				class="self-center"
				on:click={() => {
					show = false;
				}}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
					class="w-5 h-5"
				>
					<path
						d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
					/>
				</svg>
			</button>
		</div>
		<hr class=" border-gray-800" />

		<div class="flex flex-col md:flex-row w-full p-4 md:space-x-4">
			<div
				class="flex flex-row space-x-1 md:space-x-0 md:space-y-1 md:flex-col flex-1 md:flex-none md:w-40 text-gray-200 text-xs text-left mb-3 md:mb-0"
			>
				<button
					class="px-2 py-2 rounded flex-1 md:flex-none flex text-right transition {selectedMenu ===
					'general'
						? 'bg-gray-700'
						: 'hover:bg-gray-800'}"
					on:click={() => {
						selectedMenu = 'general';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M8.34 1.804A1 1 0 019.32 1h1.36a1 1 0 01.98.804l.295 1.473c.497.144.971.342 1.416.587l1.25-.834a1 1 0 011.262.125l.962.962a1 1 0 01.125 1.262l-.834 1.25c.245.445.443.919.587 1.416l1.473.294a1 1 0 01.804.98v1.361a1 1 0 01-.804.98l-1.473.295a6.95 6.95 0 01-.587 1.416l.834 1.25a1 1 0 01-.125 1.262l-.962.962a1 1 0 01-1.262.125l-1.25-.834a6.953 6.953 0 01-1.416.587l-.294 1.473a1 1 0 01-.98.804H9.32a1 1 0 01-.98-.804l-.295-1.473a6.957 6.957 0 01-1.416-.587l-1.25.834a1 1 0 01-1.262-.125l-.962-.962a1 1 0 01-.125-1.262l.834-1.25a6.957 6.957 0 01-.587-1.416l-1.473-.294A1 1 0 011 10.68V9.32a1 1 0 01.804-.98l1.473-.295c.144-.497.342-.971.587-1.416l-.834-1.25a1 1 0 01.125-1.262l.962-.962A1 1 0 015.38 3.03l1.25.834a6.957 6.957 0 011.416-.587l.294-1.473zM13 10a3 3 0 11-6 0 3 3 0 016 0z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<div class=" self-center">General</div>
				</button>

				<button
					class="px-2 py-2 rounded flex-1 md:flex-none flex text-right transition {selectedMenu ===
					'models'
						? 'bg-gray-700'
						: 'hover:bg-gray-800'}"
					on:click={() => {
						selectedMenu = 'models';
					}}
				>
					<div class=" self-center mr-2">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M10 1c3.866 0 7 1.79 7 4s-3.134 4-7 4-7-1.79-7-4 3.134-4 7-4zm5.694 8.13c.464-.264.91-.583 1.306-.952V10c0 2.21-3.134 4-7 4s-7-1.79-7-4V8.178c.396.37.842.688 1.306.953C5.838 10.006 7.854 10.5 10 10.5s4.162-.494 5.694-1.37zM3 13.179V15c0 2.21 3.134 4 7 4s7-1.79 7-4v-1.822c-.396.37-.842.688-1.306.953-1.532.875-3.548 1.369-5.694 1.369s-4.162-.494-5.694-1.37A7.009 7.009 0 013 13.179z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<div class=" self-center">Models</div>
				</button>
			</div>
			<div class="flex-1 md:min-h-[300px]">
				{#if selectedMenu === 'general'}
					<div class="flex flex-col space-y-3">
						<div>
							<div class=" mb-2.5 text-sm font-medium">Ollama Server URL</div>
							<div class="flex w-full">
								<div class="flex-1 mr-2">
									<input
										class="w-full rounded py-2 px-4 text-sm text-gray-300 bg-gray-800 outline-none"
										placeholder="Enter URL (e.g. http://localhost:11434/api)"
										bind:value={API_BASE_URL}
									/>
								</div>
								<button
									class="px-3 bg-gray-600 hover:bg-gray-700 rounded transition"
									on:click={() => {
										checkOllamaConnection();
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
											d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							</div>

							<div class="mt-2 text-xs text-gray-500">
								Trouble accessing Ollama? <a
									class=" text-gray-300 font-medium"
									href="https://github.com/ollama-webui/ollama-webui#troubleshooting"
									target="_blank"
								>
									Click here for help.
								</a>
							</div>
						</div>

						<hr class=" border-gray-700" />

						<div>
							<div class=" mb-2.5 text-sm font-medium">System Prompt</div>
							<textarea
								bind:value={system}
								class="w-full rounded p-4 text-sm text-gray-300 bg-gray-800 outline-none"
								rows="4"
							/>
						</div>

						<hr class=" border-gray-700" />

						<div>
							<label for="steps-range" class=" mb-2 text-sm font-medium flex justify-between">
								<div>Temperature</div>
								<div>
									{temperature}
								</div></label
							>
							<input
								id="steps-range"
								type="range"
								min="0"
								max="1"
								bind:value={temperature}
								step="0.05"
								class="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-700"
							/>
						</div>
						<div class="flex justify-end pt-3 text-sm font-medium">
							<button
								class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 transition rounded"
								on:click={() => {
									saveSettings(
										API_BASE_URL === '' ? BUILD_TIME_API_BASE_URL : API_BASE_URL,
										system != '' ? system : null,
										temperature != 0.8 ? temperature : null
									);
									show = false;
								}}
							>
								Save
							</button>
						</div>
					</div>
				{:else if selectedMenu === 'models'}
					<div class="flex flex-col space-y-3 text-sm">
						<div>
							<div class=" mb-2.5 text-sm font-medium">Pull a model</div>
							<div class="flex w-full">
								<div class="flex-1 mr-2">
									<input
										class="w-full rounded py-2 px-4 text-sm text-gray-300 bg-gray-800 outline-none"
										placeholder="Enter model tag (e.g. mistral:7b)"
										bind:value={modelTag}
									/>
								</div>
								<button
									class="px-3 bg-emerald-600 hover:bg-emerald-700 rounded transition"
									on:click={() => {
										pullModelHandler();
									}}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 20 20"
										fill="currentColor"
										class="w-4 h-4"
									>
										<path
											d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z"
										/>
										<path
											d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z"
										/>
									</svg>
								</button>
							</div>

							<div class="mt-2 text-xs text-gray-500">
								To access the available model names for downloading, <a
									class=" text-gray-300 font-medium"
									href="https://ollama.ai/library"
									target="_blank">click here.</a
								>
							</div>

							{#if pullProgress !== ''}
								<div class="mt-2">
									<div class=" mb-2 text-xs">Pull Progress</div>
									<div class="w-full rounded-full bg-gray-800">
										<div
											class="bg-gray-600 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded-full"
											style="width: {Math.max(15, pullProgress)}%"
										>
											{pullProgress}%
										</div>
									</div>
									<div class="mt-1 text-xs text-gray-700" style="font-size: 0.5rem;">
										{digest}
									</div>
								</div>
							{/if}
						</div>
						<hr class=" border-gray-700" />

						<div>
							<div class=" mb-2.5 text-sm font-medium">Delete a model</div>
							<div class="flex w-full">
								<div class="flex-1 mr-2">
									<input
										class="w-full rounded py-2 px-4 text-sm text-gray-300 bg-gray-800 outline-none"
										placeholder="Enter model tag (e.g. mistral:7b)"
										bind:value={deleteModelTag}
									/>
								</div>
								<button
									class="px-3 bg-red-700 hover:bg-red-800 rounded transition"
									on:click={() => {
										deleteModelHandler();
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
											d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							</div>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
</Modal>
