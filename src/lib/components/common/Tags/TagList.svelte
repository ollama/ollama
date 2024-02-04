<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();

	export let tags = [];
</script>

{#each tags as tag}
	<div
		class="px-2 py-0.5 space-x-1 flex h-fit items-center rounded-full transition border dark:border-gray-600 dark:text-white"
	>
		<div class=" text-[0.7rem] font-medium self-center line-clamp-1">
			{tag.name}
		</div>
		<button
			class=" m-auto self-center cursor-pointer"
			on:click={() => {
				dispatch('delete', tag.name);
			}}
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 16 16"
				fill="currentColor"
				class="w-3 h-3"
			>
				<path
					d="M5.28 4.22a.75.75 0 0 0-1.06 1.06L6.94 8l-2.72 2.72a.75.75 0 1 0 1.06 1.06L8 9.06l2.72 2.72a.75.75 0 1 0 1.06-1.06L9.06 8l2.72-2.72a.75.75 0 0 0-1.06-1.06L8 6.94 5.28 4.22Z"
				/>
			</svg>
		</button>
	</div>
{/each}
