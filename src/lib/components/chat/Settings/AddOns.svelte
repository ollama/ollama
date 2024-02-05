<script lang="ts">
	import toast from 'svelte-french-toast';
	import { createEventDispatcher, onMount } from 'svelte';
	import { models, voices } from '$lib/stores';
	const dispatch = createEventDispatcher();

	export let saveSettings: Function;
	// Addons
	let titleAutoGenerate = true;
	let speechAutoSend = false;
	let responseAutoCopy = false;

	let gravatarEmail = '';
	let titleAutoGenerateModel = '';

	// Voice
	let speakVoice = '';

	const toggleSpeechAutoSend = async () => {
		speechAutoSend = !speechAutoSend;
		saveSettings({ speechAutoSend: speechAutoSend });
	};

	const toggleTitleAutoGenerate = async () => {
		titleAutoGenerate = !titleAutoGenerate;
		saveSettings({ titleAutoGenerate: titleAutoGenerate });
	};

	const toggleResponseAutoCopy = async () => {
		const permission = await navigator.clipboard
			.readText()
			.then(() => {
				return 'granted';
			})
			.catch(() => {
				return '';
			});

		console.log(permission);

		if (permission === 'granted') {
			responseAutoCopy = !responseAutoCopy;
			saveSettings({ responseAutoCopy: responseAutoCopy });
		} else {
			toast.error(
				'Clipboard write permission denied. Please check your browser settings to grant the necessary access.'
			);
		}
	};

	onMount(async () => {
		let settings = JSON.parse(localStorage.getItem('settings') ?? '{}');

		titleAutoGenerate = settings.titleAutoGenerate ?? true;
		speechAutoSend = settings.speechAutoSend ?? false;
		responseAutoCopy = settings.responseAutoCopy ?? false;
		titleAutoGenerateModel = settings.titleAutoGenerateModel ?? '';
		gravatarEmail = settings.gravatarEmail ?? '';
		speakVoice = settings.speakVoice ?? '';

		const getVoicesLoop = setInterval(async () => {
			const _voices = await speechSynthesis.getVoices();
			await voices.set(_voices);

			// do your loop
			if (_voices.length > 0) {
				clearInterval(getVoicesLoop);
			}
		}, 100);
	});
</script>

<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={() => {
		saveSettings({
			speakVoice: speakVoice !== '' ? speakVoice : undefined
		});
		dispatch('save');
	}}
>
	<div class=" space-y-3">
		<div>
			<div class=" mb-1 text-sm font-medium">WebUI Add-ons</div>

			<div>
				<div class=" py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">Title Auto-Generation</div>

					<button
						class="p-1 px-3 text-xs flex rounded transition"
						on:click={() => {
							toggleTitleAutoGenerate();
						}}
						type="button"
					>
						{#if titleAutoGenerate === true}
							<span class="ml-2 self-center">On</span>
						{:else}
							<span class="ml-2 self-center">Off</span>
						{/if}
					</button>
				</div>
			</div>

			<div>
				<div class=" py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">Voice Input Auto-Send</div>

					<button
						class="p-1 px-3 text-xs flex rounded transition"
						on:click={() => {
							toggleSpeechAutoSend();
						}}
						type="button"
					>
						{#if speechAutoSend === true}
							<span class="ml-2 self-center">On</span>
						{:else}
							<span class="ml-2 self-center">Off</span>
						{/if}
					</button>
				</div>
			</div>

			<div>
				<div class=" py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">Response AutoCopy to Clipboard</div>

					<button
						class="p-1 px-3 text-xs flex rounded transition"
						on:click={() => {
							toggleResponseAutoCopy();
						}}
						type="button"
					>
						{#if responseAutoCopy === true}
							<span class="ml-2 self-center">On</span>
						{:else}
							<span class="ml-2 self-center">Off</span>
						{/if}
					</button>
				</div>
			</div>
		</div>

		<hr class=" dark:border-gray-700" />

		<div>
			<div class=" mb-2.5 text-sm font-medium">Set Title Auto-Generation Model</div>
			<div class="flex w-full">
				<div class="flex-1 mr-2">
					<select
						class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
						bind:value={titleAutoGenerateModel}
						placeholder="Select a model"
					>
						<option value="" selected>Current Model</option>
						{#each $models.filter((m) => m.size != null) as model}
							<option value={model.name} class="bg-gray-100 dark:bg-gray-700"
								>{model.name + ' (' + (model.size / 1024 ** 3).toFixed(1) + ' GB)'}</option
							>
						{/each}
					</select>
				</div>
				<button
					class="px-3 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-800 dark:text-gray-100 rounded transition"
					on:click={() => {
						saveSettings({
							titleAutoGenerateModel:
								titleAutoGenerateModel !== '' ? titleAutoGenerateModel : undefined
						});
					}}
					type="button"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 16 16"
						fill="currentColor"
						class="w-3.5 h-3.5"
					>
						<path
							fill-rule="evenodd"
							d="M13.836 2.477a.75.75 0 0 1 .75.75v3.182a.75.75 0 0 1-.75.75h-3.182a.75.75 0 0 1 0-1.5h1.37l-.84-.841a4.5 4.5 0 0 0-7.08.932.75.75 0 0 1-1.3-.75 6 6 0 0 1 9.44-1.242l.842.84V3.227a.75.75 0 0 1 .75-.75Zm-.911 7.5A.75.75 0 0 1 13.199 11a6 6 0 0 1-9.44 1.241l-.84-.84v1.371a.75.75 0 0 1-1.5 0V9.591a.75.75 0 0 1 .75-.75H5.35a.75.75 0 0 1 0 1.5H3.98l.841.841a4.5 4.5 0 0 0 7.08-.932.75.75 0 0 1 1.025-.273Z"
							clip-rule="evenodd"
						/>
					</svg>
				</button>
			</div>
		</div>

		<hr class=" dark:border-gray-700" />

		<div class=" space-y-3">
			<div>
				<div class=" mb-2.5 text-sm font-medium">Set Default Voice</div>
				<div class="flex w-full">
					<div class="flex-1">
						<select
							class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
							bind:value={speakVoice}
							placeholder="Select a voice"
						>
							<option value="" selected>Default</option>
							{#each $voices.filter((v) => v.localService === true) as voice}
								<option value={voice.name} class="bg-gray-100 dark:bg-gray-700">{voice.name}</option
								>
							{/each}
						</select>
					</div>
				</div>
			</div>
		</div>

		<!--
							<div>
								<div class=" mb-2.5 text-sm font-medium">
									Gravatar Email <span class=" text-gray-400 text-sm">(optional)</span>
								</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
											placeholder="Enter Your Email"
											bind:value={gravatarEmail}
											autocomplete="off"
											type="email"
										/>
									</div>
								</div>
								<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
									Changes user profile image to match your <a
										class=" text-gray-500 dark:text-gray-300 font-medium"
										href="https://gravatar.com/"
										target="_blank">Gravatar.</a
									>
								</div>
							</div> -->
	</div>

	<div class="flex justify-end pt-3 text-sm font-medium">
		<button
			class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
			type="submit"
		>
			Save
		</button>
	</div>
</form>
