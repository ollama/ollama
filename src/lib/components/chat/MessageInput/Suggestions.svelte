<script lang="ts">
	export let submitPrompt: Function;
	export let suggestionPrompts = [];
</script>

<div class=" flex flex-wrap-reverse mb-3 md:p-1 text-left w-full">
	{#each suggestionPrompts as prompt, promptIdx}
		<div class="{promptIdx > 1 ? 'hidden sm:inline-flex' : ''} basis-full sm:basis-1/2 p-[5px]">
			<button
				class=" flex-1 flex justify-between w-full h-full px-4 py-2.5 bg-white hover:bg-gray-50 dark:bg-gray-800 dark:hover:bg-gray-700 outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg transition group"
				on:click={() => {
					submitPrompt(prompt.content);
				}}
			>
				<div class="flex flex-col text-left self-center">
					{#if prompt.title && prompt.title[0] !== ''}
						<div class="text-sm font-medium dark:text-gray-300">{prompt.title[0]}</div>
						<div class="text-sm text-gray-500">{prompt.title[1]}</div>
					{:else}
						<div class=" self-center text-sm font-medium dark:text-gray-300 line-clamp-2">
							{prompt.content}
						</div>
					{/if}
				</div>

				<div
					class="self-center p-1 rounded-lg text-white group-hover:bg-gray-100 group-hover:text-gray-800 dark:group-hover:bg-gray-800 dark:group-hover:text-gray-300 dark:text-gray-800 transition"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 20 20"
						fill="currentColor"
						class="w-4 h-4"
					>
						<path
							fill-rule="evenodd"
							d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
							clip-rule="evenodd"
						/>
					</svg>
				</div>
			</button>
		</div>
	{/each}
</div>
