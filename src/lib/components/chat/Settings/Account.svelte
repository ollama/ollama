<script lang="ts">
	import toast from 'svelte-french-toast';
	import { onMount } from 'svelte';

	import { user } from '$lib/stores';
	import { updateUserProfile } from '$lib/apis/auths';

	import UpdatePassword from './Account/UpdatePassword.svelte';
	import { getGravatarUrl } from '$lib/apis/utils';

	export let saveHandler: Function;

	let profileImageUrl = '';
	let name = '';

	const submitHandler = async () => {
		const updatedUser = await updateUserProfile(localStorage.token, name, profileImageUrl).catch(
			(error) => {
				toast.error(error);
			}
		);

		if (updatedUser) {
			await user.set(updatedUser);
			return true;
		}
		return false;
	};

	onMount(() => {
		name = $user.name;
		profileImageUrl = $user.profile_image_url;
	});
</script>

<div class="flex flex-col h-full justify-between text-sm">
	<div class=" space-y-3 pr-1.5 overflow-y-scroll max-h-80">
		<input
			id="profile-image-input"
			type="file"
			hidden
			accept="image/*"
			on:change={(e) => {
				const files = e?.target?.files ?? [];
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
						profileImageUrl = compressedSrc;

						e.target.files = null;
					};
				};

				if (
					files.length > 0 &&
					['image/gif', 'image/jpeg', 'image/png'].includes(files[0]['type'])
				) {
					reader.readAsDataURL(files[0]);
				}
			}}
		/>

		<div class=" mb-2.5 text-sm font-medium">Profile</div>

		<div class="flex space-x-5">
			<div class="flex flex-col">
				<div class="self-center">
					<button
						class="relative rounded-full dark:bg-gray-700"
						type="button"
						on:click={() => {
							document.getElementById('profile-image-input')?.click();
						}}
					>
						<img
							src={profileImageUrl !== '' ? profileImageUrl : '/user.png'}
							alt="profile"
							class=" rounded-full w-16 h-16 object-cover"
						/>

						<div
							class="absolute flex justify-center rounded-full bottom-0 left-0 right-0 top-0 h-full w-full overflow-hidden bg-gray-700 bg-fixed opacity-0 transition duration-300 ease-in-out hover:opacity-50"
						>
							<div class="my-auto text-gray-100">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 20 20"
									fill="currentColor"
									class="w-5 h-5"
								>
									<path
										d="m2.695 14.762-1.262 3.155a.5.5 0 0 0 .65.65l3.155-1.262a4 4 0 0 0 1.343-.886L17.5 5.501a2.121 2.121 0 0 0-3-3L3.58 13.419a4 4 0 0 0-.885 1.343Z"
									/>
								</svg>
							</div>
						</div>
					</button>
				</div>
				<button
					class=" text-xs text-gray-600"
					on:click={async () => {
						const url = await getGravatarUrl($user.email);

						profileImageUrl = url;
					}}>Use Gravatar</button
				>
			</div>

			<div class="flex-1">
				<div class="flex flex-col w-full">
					<div class=" mb-1 text-xs text-gray-500">Name</div>

					<div class="flex-1">
						<input
							class="w-full rounded py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-800 outline-none"
							type="text"
							bind:value={name}
							required
						/>
					</div>
				</div>
			</div>
		</div>

		<hr class=" dark:border-gray-700 my-4" />
		<UpdatePassword />
	</div>

	<div class="flex justify-end pt-3 text-sm font-medium">
		<button
			class=" px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-gray-100 transition rounded"
			on:click={async () => {
				const res = await submitHandler();

				if (res) {
					saveHandler();
				}
			}}
		>
			Save
		</button>
	</div>
</div>
