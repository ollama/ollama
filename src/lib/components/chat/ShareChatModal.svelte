<script lang="ts">
	import Modal from '../common/Modal.svelte';

	export let downloadChat: Function;
	export let shareChat: Function;

	export let show = false;
</script>

<Modal bind:show size="xs">
	<div class="px-4 pt-4 pb-5 w-full flex flex-col justify-center">
		<button
			class=" self-center px-8 py-1.5 w-full rounded-full text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white"
			type="button"
			on:click={() => {
				shareChat();
				show = false;
			}}
		>
			Share to OllamaHub
		</button>

		<div class="flex justify-center space-x-1 mt-1.5">
			<div class=" self-center text-gray-400 text-xs font-medium">or</div>

			<button
				class=" self-center rounded-full text-xs font-medium text-gray-700 dark:text-gray-500 underline"
				type="button"
				on:click={() => {
					downloadChat();
					show = false;
				}}
			>
				Download as a File
			</button>
		</div>
	</div>
</Modal>
