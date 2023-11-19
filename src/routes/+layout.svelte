<script>
	import { onMount, tick } from 'svelte';
	import { config, user } from '$lib/stores';
	import { goto } from '$app/navigation';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import toast, { Toaster } from 'svelte-french-toast';

	import '../app.css';
	import '../tailwind.css';

	let loaded = false;

	onMount(async () => {
		const webBackendStatus = await fetch(`${WEBUI_API_BASE_URL}/`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json'
			}
		})
			.then(async (res) => {
				if (!res.ok) throw await res.json();
				return res.json();
			})
			.catch((error) => {
				console.log(error);
				return null;
			});

		console.log(webBackendStatus);
		await config.set(webBackendStatus);

		if (webBackendStatus) {
			if (webBackendStatus.auth) {
				if (localStorage.token) {
					const res = await fetch(`${WEBUI_API_BASE_URL}/auths`, {
						method: 'GET',
						headers: {
							'Content-Type': 'application/json',
							Authorization: `Bearer ${localStorage.token}`
						}
					})
						.then(async (res) => {
							if (!res.ok) throw await res.json();
							return res.json();
						})
						.catch((error) => {
							console.log(error);
							toast.error(error.detail);
							return null;
						});

					await user.set(res);
				} else {
					goto('/auth');
				}
			}
		}

		await tick();
		loaded = true;
	});
</script>

<svelte:head>
	<title>Ollama</title>
</svelte:head>
<Toaster />

{#if $config !== undefined && loaded}
	<slot />
{/if}
