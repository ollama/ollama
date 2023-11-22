<script>
	import { goto } from '$app/navigation';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import { config, user } from '$lib/stores';
	import { onMount } from 'svelte';
	import toast from 'svelte-french-toast';

	let loaded = false;
	let mode = 'signin';

	let name = '';
	let email = '';
	let password = '';

	const signInHandler = async () => {
		const res = await fetch(`${WEBUI_API_BASE_URL}/auths/signin`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				email: email,
				password: password
			})
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

		if (res) {
			console.log(res);
			toast.success(`You're now logged in.`);
			localStorage.token = res.token;
			await user.set(res);
			goto('/');
		}
	};

	const signUpHandler = async () => {
		const res = await fetch(`${WEBUI_API_BASE_URL}/auths/signup`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				name: name,
				email: email,
				password: password
			})
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

		if (res) {
			console.log(res);
			toast.success(`Account creation successful."`);
			localStorage.token = res.token;
			await user.set(res);
			goto('/');
		}
	};

	onMount(async () => {
		if ($config === null || !$config.auth || ($config.auth && $user !== undefined)) {
			await goto('/');
		}
		loaded = true;
	});
</script>

{#if loaded && $config && $config.auth}
	<div class="fixed m-10 z-50">
		<div class="flex space-x-2">
			<div class=" self-center">
				<img src="/ollama.png" class=" w-8" />
			</div>
		</div>
	</div>

	<div class=" bg-white min-h-screen w-full flex justify-center font-mona">
		<div class="hidden lg:flex lg:flex-1 px-10 md:px-16 w-full bg-yellow-50 justify-center">
			<div class=" my-auto pb-16 text-left">
				<div>
					<div class=" font-bold text-yellow-600 text-4xl">
						Get up and running with <br />large language models, locally.
					</div>

					<div class="mt-2 text-yellow-600 text-xl">
						Run Llama 2, Code Llama, and other models. Customize and create your own.
					</div>
				</div>
			</div>
		</div>

		<div class="w-full max-w-xl px-10 md:px-16 bg-white min-h-screen w-full flex flex-col">
			<div class=" my-auto pb-10 w-full">
				<form
					class=" flex flex-col justify-center"
					on:submit|preventDefault={() => {
						if (mode === 'signin') {
							signInHandler();
						} else {
							signUpHandler();
						}
					}}
				>
					<div class=" text-2xl md:text-3xl font-semibold">
						{mode === 'signin' ? 'Sign in' : 'Sign up'} to Ollama Web UI
					</div>

					<hr class="my-8" />

					<div class="flex flex-col space-y-4">
						{#if mode === 'signup'}
							<div>
								<div class=" text-sm font-bold text-left mb-2">Name</div>
								<input
									bind:value={name}
									type="text"
									class=" border px-5 py-4 rounded-2xl w-full text-sm"
									autocomplete="name"
									required
								/>
							</div>
						{/if}

						<div>
							<div class=" text-sm font-bold text-left mb-2">Email</div>
							<input
								bind:value={email}
								type="email"
								class=" border px-5 py-4 rounded-2xl w-full text-sm"
								autocomplete="email"
								required
							/>
						</div>

						<div>
							<div class=" text-sm font-bold text-left mb-2">Password</div>
							<input
								bind:value={password}
								type="password"
								class=" border px-5 py-4 rounded-2xl w-full text-sm"
								autocomplete="current-password"
								required
							/>
						</div>
					</div>

					<div class="mt-8">
						<button
							class=" bg-gray-900 hover:bg-gray-800 w-full rounded-full text-white font-semibold text-sm py-5 transition"
							type="submit"
						>
							{mode === 'signin' ? 'Sign In' : 'Create Account'}
						</button>

						<div class=" mt-4 text-sm text-center">
							{mode === 'signin' ? `Don't have an account?` : `Already have an account?`}

							<button
								class=" font-medium underline"
								type="button"
								on:click={() => {
									if (mode === 'signin') {
										mode = 'signup';
									} else {
										mode = 'signin';
									}
								}}
							>
								{mode === 'signin' ? `Sign up` : `Sign In`}
							</button>
						</div>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}

<style>
	.font-mona {
		font-family: 'Mona Sans';
	}
</style>
