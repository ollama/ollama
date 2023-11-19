<script>
	import { config, user } from '$lib/stores';
	import { goto } from '$app/navigation';

	if ($config && $config.auth && $user === undefined) {
		goto('/auth');
	}
</script>

{#if $config !== undefined}
	<slot />
{/if}
