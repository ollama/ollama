<script>
	import { onMount, tick } from 'svelte';
	import { config, user } from '$lib/stores';
	import { goto } from '$app/navigation';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import toast, { Toaster } from 'svelte-french-toast';

	import '../app.css';
	import '../tailwind.css';
	import 'tippy.js/dist/tippy.css';
	let loaded = false;

	onMount(async () => {
		// Check Backend Status
		const res = await fetch(`${WEBUI_API_BASE_URL}/`, {
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

		if (res) {
			await config.set(res);
			console.log(res);

			if ($config) {
				if (localStorage.token) {
					// Get Session User Info
					const sessionUser = await fetch(`${WEBUI_API_BASE_URL}/auths`, {
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
