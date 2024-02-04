<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();

	let showTagInput = false;
	let tagName = '';
</script>

<div class="flex space-x-1 pl-1.5">
	{#if showTagInput}
		<div class="flex items-center">
			<input
				bind:value={tagName}
				class=" cursor-pointer self-center text-xs h-fit bg-transparent outline-none line-clamp-1 w-[4rem]"
				placeholder="Add a tag"
			/>

			<button
				type="button"
				on:click={() => {
					dispatch('add', tagName);
					tagName = '';
					showTagInput = false;
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
						d="M12.416 3.376a.75.75 0 0 1 .208 1.04l-5 7.5a.75.75 0 0 1-1.154.114l-3-3a.75.75 0 0 1 1.06-1.06l2.353 2.353 4.493-6.74a.75.75 0 0 1 1.04-.207Z"
						clip-rule="evenodd"
					/>
				</svg>
			</button>
		</div>

		<!-- TODO: Tag Suggestions -->
	{/if}

	<button
		class=" cursor-pointer self-center p-0.5 space-x-1 flex h-fit items-center dark:hover:bg-gray-700 rounded-full transition border dark:border-gray-600 border-dashed"
		type="button"
		on:click={() => {
			showTagInput = !showTagInput;
		}}
	>
		<div class=" m-auto self-center">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 16 16"
				fill="currentColor"
				class="w-3 h-3 {showTagInput ? 'rotate-45' : ''} transition-all transform"
			>
				<path
					d="M8.75 3.75a.75.75 0 0 0-1.5 0v3.5h-3.5a.75.75 0 0 0 0 1.5h3.5v3.5a.75.75 0 0 0 1.5 0v-3.5h3.5a.75.75 0 0 0 0-1.5h-3.5v-3.5Z"
				/>
			</svg>
		</div>
	</button>
</div>
