<script lang="ts">
	import { modelfiles } from '$lib/stores';
	import { onMount } from 'svelte';

	onMount(() => {});
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
				<a
					class=" flex space-x-4 cursor-pointer w-full mb-3"
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
						<div class=" font-bold">{modelfile.title}</div>
						<div class=" text-sm overflow-hidden text-ellipsis line-clamp-2">{modelfile.desc}</div>
					</div>
				</a>
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
