<script>
	import { goto } from '$app/navigation';
	import { config } from '$lib/stores';
	import { onMount } from 'svelte';

	let loaded = false;

	onMount(async () => {
		if ($config) {
			await goto('/');
		}

		loaded = true;
	});
</script>

{#if loaded}
	<div class="absolute w-full h-full flex z-50">
		<div class="absolute rounded-xl w-full h-full backdrop-blur bg-gray-900/5 flex justify-center">
			<div class="m-auto pb-44 flex flex-col justify-center">
				<div class="max-w-md">
					<div class="text-center text-2xl font-medium z-50">Ollama WebUI Backend Required</div>

					<div class=" mt-4 text-center text-sm w-full">
						Oops! It seems like your Ollama WebUI needs a little attention. <br
							class=" hidden sm:flex"
						/>
						describe troubleshooting/installation, help @ discord

						<!-- TODO: update text -->
					</div>

					<div class=" mt-6 mx-auto relative group w-fit">
						<button
							class="relative z-20 flex px-5 py-2 rounded-full bg-gray-100 hover:bg-gray-200 transition font-medium text-sm"
						>
							Check Again
						</button>
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}
