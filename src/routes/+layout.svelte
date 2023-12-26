<script>
	import { onMount, tick } from 'svelte';
	import { config, user } from '$lib/stores';
	import { goto } from '$app/navigation';
	import toast, { Toaster } from 'svelte-french-toast';

	import { getBackendConfig } from '$lib/apis';
	import { getSessionUser } from '$lib/apis/auths';

	import '../app.css';
	import '../tailwind.css';
	import 'tippy.js/dist/tippy.css';

	let loaded = false;

	onMount(async () => {
		// Check Backend Status
		const backendConfig = await getBackendConfig();

		if (backendConfig) {
			await config.set(backendConfig);
			console.log(backendConfig);

			if ($config) {
				if (localStorage.token) {
					// Get Session User Info
					const sessionUser = await getSessionUser(localStorage.token).catch((error) => {
						toast.error(error);
						return null;
					});

					if (sessionUser) {
						await user.set(sessionUser);
					} else {
						localStorage.removeItem('token');
						await goto('/auth');
					}
				} else {
					await goto('/auth');
				}
			}
		} else {
			await goto(`/error`);
		}

		await tick();
		loaded = true;
	});
</script>

<svelte:head>
	<title>Ollama</title>
</svelte:head>

{#if loaded}
	<slot />
{/if}

<Toaster />
