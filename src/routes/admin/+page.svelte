<script>
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import { config, user } from '$lib/stores';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	import toast from 'svelte-french-toast';

	let loaded = false;
	let users = [];

	const updateUserRole = async (id, role) => {
		const res = await fetch(`${WEBUI_API_BASE_URL}/users/update/role`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: `Bearer ${localStorage.token}`
			},
			body: JSON.stringify({
				id: id,
				role: role
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
			await getUsers();
		}
	};

	const getUsers = async () => {
		const res = await fetch(`${WEBUI_API_BASE_URL}/users/`, {
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

		users = res ? res : [];
	};

	onMount(async () => {
		if ($config === null || !$config.auth || ($config.auth && $user && $user.role !== 'admin')) {
			await goto('/');
		} else {
			await getUsers();
		}
		loaded = true;
	});
</script>

<div
	class=" bg-white dark:bg-gray-800 dark:text-gray-100 min-h-screen w-full flex justify-center font-mona"
>
	{#if loaded}
		<div class="w-full max-w-3xl px-10 md:px-16 min-h-screen flex flex-col">
			<div class="py-10 w-full">
				<div class=" flex flex-col justify-center">
					<div class=" text-2xl font-semibold">Users ({users.length})</div>
					<div class=" text-gray-500 text-xs font-medium mt-1">
						Click on the user role cell in the table to change a user's role.
					</div>

					<hr class=" my-3 dark:border-gray-600" />

					<div class="scrollbar-hidden relative overflow-x-auto whitespace-nowrap">
						<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400">
							<thead
								class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400"
							>
								<tr>
									<th scope="col" class="px-6 py-3"> Name </th>
									<th scope="col" class="px-6 py-3"> Email </th>
									<th scope="col" class="px-6 py-3"> Role </th>
									<!-- <th scope="col" class="px-6 py-3"> Action </th> -->
								</tr>
							</thead>
							<tbody>
								{#each users as user}
									<tr class="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
										<th
											scope="row"
											class="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white"
										>
											<div class="flex flex-row">
												<img
													class=" rounded-full max-w-[30px] max-h-[30px] object-cover mr-4"
													src={user.profile_image_url}
												/>

												<div class=" font-semibold md:self-center">{user.name}</div>
											</div>
										</th>
										<td class="px-6 py-4"> {user.email} </td>
										<td class="px-6 py-4">
											<button
												class="  dark:text-white underline"
												on:click={() => {
													if (user.role === 'user') {
														updateUserRole(user.id, 'admin');
													} else if (user.role === 'pending') {
														updateUserRole(user.id, 'user');
													} else {
														updateUserRole(user.id, 'pending');
													}
												}}>{user.role}</button
											>
										</td>
										<!-- <td class="px-6 py-4 text-center">
											<button class="  text-white underline"> Edit </button>
										</td> -->
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.font-mona {
		font-family: 'Mona Sans';
	}

	.scrollbar-hidden::-webkit-scrollbar {
		display: none; /* for Chrome, Safari and Opera */
	}

	.scrollbar-hidden {
		-ms-overflow-style: none; /* IE and Edge */
		scrollbar-width: none; /* Firefox */
	}
</style>
