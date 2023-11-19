<script>
	import { config, user } from '$lib/stores';
	import { goto } from '$app/navigation';
	import { onMount, tick } from 'svelte';

	let loaded = false;
	onMount(async () => {
		if ($config && $config.auth && $user === undefined) {
			await goto('/auth');
		}

		await tick();
		loaded = true;
	});
</script>

{#if loaded}
	<slot />
{/if}
