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
					<div class=" self-center">
						<a
							class=" w-fit text-sm px-3 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							href={`/modelfiles/edit?tag=${modelfile.tagName}`}
						>
							Edit</a
						>

						<button
							class=" w-fit text-sm px-3 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							on:click={() => {
								deleteModelfilebyTagName(modelfile.tagName);
							}}
						>
							Delete</button
						>
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
