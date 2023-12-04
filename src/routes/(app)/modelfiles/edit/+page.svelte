<script>
	import { v4 as uuidv4 } from 'uuid';
	import { toast } from 'svelte-french-toast';
	import { goto } from '$app/navigation';
	import { OLLAMA_API_BASE_URL } from '$lib/constants';
	import { settings, db, user, config, modelfiles } from '$lib/stores';

	import Advanced from '$lib/components/chat/Settings/Advanced.svelte';
	import { splitStream } from '$lib/utils';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';

	let loading = false;

	let filesInputElement;
	let inputFiles;
	let imageUrl = null;
	let digest = '';
	let pullProgress = null;
	let success = false;

	let modelfile = null;
	// ///////////
	// Modelfile
	// ///////////

	let title = '';
	let tagName = '';
	let desc = '';

	// Raw Mode
	let content = '';

	let suggestions = [
		{
			content: ''
		}
	];

	let categories = {
		character: false,
		assistant: false,
		writing: false,
		productivity: false,
		programming: false,
		'data analysis': false,
		lifestyle: false,
		education: false,
		business: false
	};

	onMount(() => {
		tagName = $page.url.searchParams.get('tag');

		if (tagName) {
			modelfile = $modelfiles.filter((modelfile) => modelfile.tagName === tagName)[0];

			console.log(modelfile);

			imageUrl = modelfile.imageUrl;
			title = modelfile.title;
			desc = modelfile.desc;
			content = modelfile.content;
			suggestions =
				modelfile.suggestionPrompts.length != 0
					? modelfile.suggestionPrompts
					: [
							{
								content: ''
							}
					  ];

			for (const category of modelfile.categories) {
				categories[category.toLowerCase()] = true;
			}
		} else {
			goto('/modelfiles');
		}
	});

	const saveModelfile = async (modelfile) => {
		await modelfiles.set(
			$modelfiles.map((e) => {
				if (e.tagName === modelfile.tagName) {
					return modelfile;
				} else {
					return e;
				}
			})
		);
		localStorage.setItem('modelfiles', JSON.stringify($modelfiles));
	};

	const updateHandler = async () => {
		loading = true;

		if (Object.keys(categories).filter((category) => categories[category]).length == 0) {
			toast.error(
				'Uh-oh! It looks like you missed selecting a category. Please choose one to complete your modelfile.'
			);
		}

		if (
			title !== '' &&
			desc !== '' &&
			content !== '' &&
			Object.keys(categories).filter((category) => categories[category]).length > 0
		) {
			const res = await fetch(`${$settings?.API_BASE_URL ?? OLLAMA_API_BASE_URL}/create`, {
				method: 'POST',
				headers: {
					'Content-Type': 'text/event-stream',
					...($settings.authHeader && { Authorization: $settings.authHeader }),
					...($user && { Authorization: `Bearer ${localStorage.token}` })
				},
				body: JSON.stringify({
					name: tagName,
					modelfile: content
				})
			});

			if (res) {
				const reader = res.body
					.pipeThrough(new TextDecoderStream())
					.pipeThrough(splitStream('\n'))
					.getReader();

				while (true) {
					const { value, done } = await reader.read();
					if (done) break;

					try {
						let lines = value.split('\n');

						for (const line of lines) {
							if (line !== '') {
								console.log(line);
								let data = JSON.parse(line);
								console.log(data);

								if (data.error) {
									throw data.error;
								}
								if (data.detail) {
									throw data.detail;
								}

								if (data.status) {
									if (
										!data.digest &&
										!data.status.includes('writing') &&
										!data.status.includes('sha256')
									) {
										toast.success(data.status);

										if (data.status === 'success') {
											success = true;
										}
									} else {
										if (data.digest) {
											digest = data.digest;

											if (data.completed) {
												pullProgress = Math.round((data.completed / data.total) * 1000) / 10;
											} else {
												pullProgress = 100;
											}
										}
									}
								}
							}
						}
					} catch (error) {
						console.log(error);
						toast.error(error);
					}
				}
			}

			if (success) {
				await saveModelfile({
					tagName: tagName,
					imageUrl: imageUrl,
					title: title,
					desc: desc,
					content: content,
					suggestionPrompts: suggestions.filter((prompt) => prompt.content !== ''),
					categories: Object.keys(categories).filter((category) => categories[category])
				});
				await goto('/modelfiles');
			}
		}
		loading = false;
		success = false;
	};
</script>

<div class="min-h-screen w-full flex justify-center dark:text-white">
	<div class=" py-2.5 flex flex-col justify-between w-full">
		<div class="max-w-2xl mx-auto w-full px-3 md:px-0 my-10">
			<input
				bind:this={filesInputElement}
				bind:files={inputFiles}
				type="file"
				hidden
				accept="image/*"
				on:change={() => {
					let reader = new FileReader();
					reader.onload = (event) => {
						let originalImageUrl = `${event.target.result}`;

						const img = new Image();
						img.src = originalImageUrl;

						img.onload = function () {
							const canvas = document.createElement('canvas');
							const ctx = canvas.getContext('2d');

							// Calculate the aspect ratio of the image
							const aspectRatio = img.width / img.height;

							// Calculate the new width and height to fit within 100x100
							let newWidth, newHeight;
							if (aspectRatio > 1) {
								newWidth = 100 * aspectRatio;
								newHeight = 100;
							} else {
								newWidth = 100;
								newHeight = 100 / aspectRatio;
							}

							// Set the canvas size
							canvas.width = 100;
							canvas.height = 100;

							// Calculate the position to center the image
							const offsetX = (100 - newWidth) / 2;
							const offsetY = (100 - newHeight) / 2;

							// Draw the image on the canvas
							ctx.drawImage(img, offsetX, offsetY, newWidth, newHeight);

							// Get the base64 representation of the compressed image
							const compressedSrc = canvas.toDataURL('image/jpeg');

							// Display the compressed image
							imageUrl = compressedSrc;

							inputFiles = null;
						};
					};

					if (
						inputFiles &&
						inputFiles.length > 0 &&
						['image/gif', 'image/jpeg', 'image/png'].includes(inputFiles[0]['type'])
					) {
						reader.readAsDataURL(inputFiles[0]);
					} else {
						console.log(`Unsupported File Type '${inputFiles[0]['type']}'.`);
						inputFiles = null;
					}
				}}
			/>

			<div class=" text-2xl font-semibold mb-6">My Modelfiles</div>

			<button
				class="flex space-x-1"
				on:click={() => {
					history.back();
				}}
			>
				<div class=" self-center">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 20 20"
						fill="currentColor"
						class="w-4 h-4"
					>
						<path
							fill-rule="evenodd"
							d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z"
							clip-rule="evenodd"
						/>
					</svg>
				</div>
				<div class=" self-center font-medium text-sm">Back</div>
			</button>
			<hr class="my-3 dark:border-gray-700" />

			<form
				class="flex flex-col"
				on:submit|preventDefault={() => {
					updateHandler();
				}}
			>
				<div class="flex justify-center my-4">
					<div class="self-center">
						<button
							class=" {imageUrl
								? ''
								: 'p-6'} rounded-full dark:bg-gray-700 border border-dashed border-gray-200"
							type="button"
							on:click={() => {
								filesInputElement.click();
							}}
						>
							{#if imageUrl}
								<img
									src={imageUrl}
									alt="modelfile profile"
									class=" rounded-full w-20 h-20 object-cover"
								/>
							{:else}
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 24 24"
									fill="currentColor"
									class="w-8"
								>
									<path
										fill-rule="evenodd"
										d="M12 3.75a.75.75 0 01.75.75v6.75h6.75a.75.75 0 010 1.5h-6.75v6.75a.75.75 0 01-1.5 0v-6.75H4.5a.75.75 0 010-1.5h6.75V4.5a.75.75 0 01.75-.75z"
										clip-rule="evenodd"
									/>
								</svg>
							{/if}
						</button>
					</div>
				</div>

				<div class="my-2 flex space-x-2">
					<div class="flex-1">
						<div class=" text-sm font-semibold mb-2">Name*</div>

						<div>
							<input
								class="px-3 py-1.5 text-sm w-full bg-transparent border dark:border-gray-600 outline-none rounded-lg"
								placeholder="Name your modelfile"
								bind:value={title}
								required
							/>
						</div>
					</div>

					<div class="flex-1">
						<div class=" text-sm font-semibold mb-2">Model Tag Name*</div>

						<div>
							<input
								class="px-3 py-1.5 text-sm w-full bg-transparent disabled:text-gray-500 border dark:border-gray-600 outline-none rounded-lg"
								placeholder="Add a model tag name"
								value={tagName}
								disabled
								required
							/>
						</div>
					</div>
				</div>

				<div class="my-2">
					<div class=" text-sm font-semibold mb-2">Description*</div>

					<div>
						<input
							class="px-3 py-1.5 text-sm w-full bg-transparent border dark:border-gray-600 outline-none rounded-lg"
							placeholder="Add a short description about what this modelfile does"
							bind:value={desc}
							required
						/>
					</div>
				</div>

				<div class="my-2">
					<div class="flex w-full justify-between">
						<div class=" self-center text-sm font-semibold">Modelfile</div>
					</div>

					<!-- <div class=" text-sm font-semibold mb-2"></div> -->

					<div class="mt-2">
						<div class=" text-xs font-semibold mb-2">Content*</div>

						<div>
							<textarea
								class="px-3 py-1.5 text-sm w-full bg-transparent border dark:border-gray-600 outline-none rounded-lg"
								placeholder={`FROM llama2\nPARAMETER temperature 1\nSYSTEM """\nYou are Mario from Super Mario Bros, acting as an assistant.\n"""`}
								rows="6"
								bind:value={content}
								required
							/>
						</div>
					</div>
				</div>

				<div class="my-2">
					<div class="flex w-full justify-between mb-2">
						<div class=" self-center text-sm font-semibold">Prompt suggestions</div>

						<button
							class="p-1 px-3 text-xs flex rounded transition"
							type="button"
							on:click={() => {
								if (suggestions.length === 0 || suggestions.at(-1).content !== '') {
									suggestions = [...suggestions, { content: '' }];
								}
							}}
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 20 20"
								fill="currentColor"
								class="w-4 h-4"
							>
								<path
									d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z"
								/>
							</svg>
						</button>
					</div>
					<div class="flex flex-col space-y-1">
						{#each suggestions as prompt, promptIdx}
							<div class=" flex border dark:border-gray-600 rounded-lg">
								<input
									class="px-3 py-1.5 text-sm w-full bg-transparent outline-none border-r dark:border-gray-600"
									placeholder="Write a prompt suggestion (e.g. Who are you?)"
									bind:value={prompt.content}
								/>

								<button
									class="px-2"
									type="button"
									on:click={() => {
										suggestions.splice(promptIdx, 1);
										suggestions = suggestions;
									}}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 20 20"
										fill="currentColor"
										class="w-4 h-4"
									>
										<path
											d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
										/>
									</svg>
								</button>
							</div>
						{/each}
					</div>
				</div>

				<div class="my-2">
					<div class=" text-sm font-semibold mb-2">Categories</div>

					<div class="grid grid-cols-4">
						{#each Object.keys(categories) as category}
							<div class="flex space-x-2 text-sm">
								<input type="checkbox" bind:checked={categories[category]} />

								<div class=" capitalize">{category}</div>
							</div>
						{/each}
					</div>
				</div>

				{#if pullProgress !== null}
					<div class="my-2">
						<div class=" text-sm font-semibold mb-2">Pull Progress</div>
						<div class="w-full rounded-full dark:bg-gray-800">
							<div
								class="dark:bg-gray-600 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded-full"
								style="width: {Math.max(15, pullProgress ?? 0)}%"
							>
								{pullProgress ?? 0}%
							</div>
						</div>
						<div class="mt-1 text-xs dark:text-gray-500" style="font-size: 0.5rem;">
							{digest}
						</div>
					</div>
				{/if}

				<div class="my-2 flex justify-end">
					<button
						class=" text-sm px-3 py-2 transition rounded-xl {loading
							? ' cursor-not-allowed bg-gray-100 dark:bg-gray-800'
							: ' bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-800'} flex"
						type="submit"
						disabled={loading}
					>
						<div class=" self-center font-medium">Save & Update</div>

						{#if loading}
							<div class="ml-1.5 self-center">
								<svg
									class=" w-4 h-4"
									viewBox="0 0 24 24"
									fill="currentColor"
									xmlns="http://www.w3.org/2000/svg"
									><style>
										.spinner_ajPY {
											transform-origin: center;
											animation: spinner_AtaB 0.75s infinite linear;
										}
										@keyframes spinner_AtaB {
											100% {
												transform: rotate(360deg);
											}
										}
									</style><path
										d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z"
										opacity=".25"
									/><path
										d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z"
										class="spinner_ajPY"
									/></svg
								>
							</div>
						{/if}
					</button>
				</div>
			</form>
		</div>
	</div>
</div>
