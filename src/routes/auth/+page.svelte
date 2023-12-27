<script>
	import { goto } from '$app/navigation';
	import { userSignIn, userSignUp } from '$lib/apis/auths';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import { config, user } from '$lib/stores';
	import { onMount } from 'svelte';
	import toast from 'svelte-french-toast';

	let loaded = false;
	let mode = 'signin';

	let name = '';
	let email = '';
	let password = '';

	const setSessionUser = async (sessionUser) => {
		if (sessionUser) {
			console.log(sessionUser);
			toast.success(`You're now logged in.`);
			localStorage.token = sessionUser.token;
			await user.set(sessionUser);
			goto('/');
		}
	};

	const signInHandler = async () => {
		const sessionUser = await userSignIn(email, password).catch((error) => {
			toast.error(error);
			return null;
		});

		await setSessionUser(sessionUser);
	};

	const signUpHandler = async () => {
		const sessionUser = await userSignUp(name, email, password).catch((error) => {
			toast.error(error);
			return null;
		});

		await setSessionUser(sessionUser);
	};

	const submitHandler = async () => {
		if (mode === 'signin') {
			await signInHandler();
		} else {
			await signUpHandler();
		}
	};

	onMount(async () => {
		if ($user !== undefined) {
			await goto('/');
		}
		loaded = true;
	});
</script>

{#if loaded}
	<div class="fixed m-10 z-50">
		<div class="flex space-x-2">
			<div class=" self-center">
				<img src="/ollama.png" class=" w-8" />
			</div>
		</div>
	</div>

	<div class=" bg-white min-h-screen w-full flex justify-center font-mona">
		<!-- <div class="hidden lg:flex lg:flex-1 px-10 md:px-16 w-full bg-yellow-50 justify-center">
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
		</div> -->

		<div class="w-full max-w-lg px-10 md:px-16 bg-white min-h-screen flex flex-col">
			<div class=" my-auto pb-10 w-full">
				<form
					class=" flex flex-col justify-center"
					on:submit|preventDefault={() => {
						submitHandler();
					}}
				>
					<div class=" text-xl md:text-2xl font-bold">
						{mode === 'signin' ? 'Sign in' : 'Sign up'} to Ollama Web UI
					</div>

					<div class="flex flex-col mt-4">
						{#if mode === 'signup'}
							<div>
								<div class=" text-sm font-semibold text-left mb-1">Name</div>
								<input
									bind:value={name}
									type="text"
									class=" border px-4 py-2.5 rounded-2xl w-full text-sm"
									autocomplete="name"
									placeholder="Enter Your Full Name"
									required
								/>
							</div>

							<hr class=" my-3" />
						{/if}

						<div class="mb-2">
							<div class=" text-sm font-semibold text-left mb-1">Email</div>
							<input
								bind:value={email}
								type="email"
								class=" border px-4 py-2.5 rounded-2xl w-full text-sm"
								autocomplete="email"
								placeholder="Enter Your Email"
								required
							/>
						</div>

						<div>
							<div class=" text-sm font-semibold text-left mb-1">Password</div>
							<input
								bind:value={password}
								type="password"
								class=" border px-4 py-2.5 rounded-2xl w-full text-sm"
								placeholder="Enter Your Password"
								autocomplete="current-password"
								required
							/>
						</div>
					</div>

					<div class="mt-5">
						<button
							class=" bg-gray-900 hover:bg-gray-800 w-full rounded-full text-white font-semibold text-sm py-3 transition"
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
