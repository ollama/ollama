<script lang="ts">
	import { modelfiles, settings, user } from '$lib/stores';
	import { onMount } from 'svelte';
	import toast from 'svelte-french-toast';

	import { OLLAMA_API_BASE_URL } from '$lib/constants';

	const deleteModelHandler = async (tagName) => {
		let success = null;
		const res = await fetch(`${$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/delete`, {
			method: 'DELETE',
			headers: {
				'Content-Type': 'text/event-stream',
				...($settings.authHeader && { Authorization: $settings.authHeader }),
				...($user && { Authorization: `Bearer ${localStorage.token}` })
			},
			body: JSON.stringify({
				name: tagName
			})
		})
			.then(async (res) => {
				if (!res.ok) throw await res.json();
				return res.json();
			})
			.then((json) => {
				console.log(json);
				toast.success(`Deleted ${tagName}`);
				success = true;
				return json;
			})
			.catch((err) => {
				console.log(err);
				toast.error(err.error);
				return null;
			});

		return success;
	};

	const deleteModelfilebyTagName = async (tagName) => {
		await deleteModelHandler(tagName);
		await modelfiles.set($modelfiles.filter((modelfile) => modelfile.tagName != tagName));
		localStorage.setItem('modelfiles', JSON.stringify($modelfiles));
	};

	const shareModelfile = async (modelfile) => {
		toast.success('Redirecting you to OllamaHub');

		const url = 'https://ollamahub.com';

		const tab = await window.open(`${url}/create`, '_blank');
		window.addEventListener(
			'message',
			(event) => {
				if (event.origin !== url) return;
				if (event.data === 'loaded') {
					tab.postMessage(JSON.stringify(modelfile), '*');
				}
			},
			false
		);
	};
</script>

<div class="min-h-screen w-full flex justify-center dark:text-white">
	<div class=" py-2.5 flex flex-col justify-between w-full">
		<div class="max-w-2xl mx-auto w-full px-3 md:px-0 my-10">
			<div class=" text-2xl font-semibold mb-6">My Modelfiles</div>

			<a class=" flex space-x-4 cursor-pointer w-full mb-3" href="/modelfiles/create">
				<div class=" self-center w-10">
					<div
						class="w-full h-10 flex justify-center rounded-full bg-transparent dark:bg-gray-700 border border-dashed border-gray-200"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 24 24"
							fill="currentColor"
							class="w-6"
						>
							<path
								fill-rule="evenodd"
								d="M12 3.75a.75.75 0 01.75.75v6.75h6.75a.75.75 0 010 1.5h-6.75v6.75a.75.75 0 01-1.5 0v-6.75H4.5a.75.75 0 010-1.5h6.75V4.5a.75.75 0 01.75-.75z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
				</div>

				<div class=" self-center">
					<div class=" font-bold">Create a modelfile</div>
					<div class=" text-sm">Customize Ollama models for a specific purpose</div>
				</div>
			</a>

			{#each $modelfiles as modelfile}
				<hr class=" dark:border-gray-700 my-2.5" />
				<div class=" flex space-x-4 cursor-pointer w-full mb-3">
					<a
						class=" flex flex-1 space-x-4 cursor-pointer w-full"
						href={`/?models=${modelfile.tagName}`}
					>
						<div class=" self-center w-10">
							<div class=" rounded-full bg-stone-700">
								<img
									src={modelfile.imageUrl ?? '/user.png'}
									alt="modelfile profile"
									class=" rounded-full w-full h-auto object-cover"
								/>
							</div>
						</div>

						<div class=" flex-1 self-center">
							<div class=" font-bold capitalize">{modelfile.title}</div>
							<div class=" text-sm overflow-hidden text-ellipsis line-clamp-2">
								{modelfile.desc}
							</div>
						</div>
					</a>
					<div class="flex flex-row space-x-1 self-center">
						<a
							class="self-center w-fit text-sm px-2 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							href={`/modelfiles/edit?tag=${modelfile.tagName}`}
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
						</a>

						<button
							class="self-center w-fit text-sm px-2 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							on:click={() => {
								shareModelfile(modelfile);
							}}
						>
							<!-- TODO: update to share icon -->
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
									d="M7.217 10.907a2.25 2.25 0 100 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186l9.566-5.314m-9.566 7.5l9.566 5.314m0 0a2.25 2.25 0 103.935 2.186 2.25 2.25 0 00-3.935-2.186zm0-12.814a2.25 2.25 0 103.933-2.185 2.25 2.25 0 00-3.933 2.185z"
								/>
							</svg>
						</button>

						<button
							class="self-center w-fit text-sm px-2 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							on:click={() => {
								deleteModelfilebyTagName(modelfile.tagName);
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
				</div>
			{/each}

			<div class=" my-16">
				<div class=" text-2xl font-semibold mb-6">Made by OllamaHub Community</div>

				<a
					class=" flex space-x-4 cursor-pointer w-full mb-3"
					href="https://ollamahub.com/"
					target="_blank"
				>
					<div class=" self-center w-10">
						<div
							class="w-full h-10 flex justify-center rounded-full bg-transparent dark:bg-gray-700 border border-dashed border-gray-200"
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 24 24"
								fill="currentColor"
								class="w-6"
							>
								<path
									fill-rule="evenodd"
									d="M12 3.75a.75.75 0 01.75.75v6.75h6.75a.75.75 0 010 1.5h-6.75v6.75a.75.75 0 01-1.5 0v-6.75H4.5a.75.75 0 010-1.5h6.75V4.5a.75.75 0 01.75-.75z"
									clip-rule="evenodd"
								/>
							</svg>
						</div>
					</div>

					<div class=" self-center">
						<div class=" font-bold">Discover a modelfile</div>
						<div class=" text-sm">Discover, download, and explore Ollama Modelfiles</div>
					</div>
				</a>
			</div>
		</div>
	</div>
</div>
