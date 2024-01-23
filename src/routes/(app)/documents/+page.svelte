<script lang="ts">
	import toast from 'svelte-french-toast';
	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { onMount } from 'svelte';
	import { documents } from '$lib/stores';
	import { createNewDoc, deleteDocByName, getDocs } from '$lib/apis/documents';

	import { SUPPORTED_FILE_TYPE, SUPPORTED_FILE_EXTENSIONS } from '$lib/constants';
	import { uploadDocToVectorDB } from '$lib/apis/rag';
	import { transformFileName } from '$lib/utils';

	import EditDocModal from '$lib/components/documents/EditDocModal.svelte';

	let importFiles = '';

	let inputFiles = '';
	let query = '';

	let showEditDocModal = false;
	let selectedDoc;

	let dragged = false;

	const deleteDoc = async (name) => {
		await deleteDocByName(localStorage.token, name);
		await documents.set(await getDocs(localStorage.token));
	};

	const uploadDoc = async (file) => {
		const res = await uploadDocToVectorDB(localStorage.token, '', file).catch((error) => {
			toast.error(error);
			return null;
		});

		if (res) {
			await createNewDoc(
				localStorage.token,
				res.collection_name,
				res.filename,
				transformFileName(res.filename),
				res.filename
			).catch((error) => {
				toast.error(error);
				return null;
			});
			await documents.set(await getDocs(localStorage.token));
		}
	};

	const onDragOver = (e) => {
		e.preventDefault();
		dragged = true;
	};

	const onDragLeave = () => {
		dragged = false;
	};

	const onDrop = async (e) => {
		e.preventDefault();
		console.log(e);

		if (e.dataTransfer?.files) {
			const inputFiles = e.dataTransfer?.files;

			if (inputFiles && inputFiles.length > 0) {
				const file = inputFiles[0];
				if (
					SUPPORTED_FILE_TYPE.includes(file['type']) ||
					SUPPORTED_FILE_EXTENSIONS.includes(file.name.split('.').at(-1))
				) {
					uploadDoc(file);
				} else {
					toast.error(
						`Unknown File Type '${file['type']}', but accepting and treating as plain text`
					);
					uploadDoc(file);
				}
			} else {
				toast.error(`File not found.`);
			}
		}

		dragged = false;
	};
</script>

{#key selectedDoc}
	<EditDocModal bind:show={showEditDocModal} {selectedDoc} />
{/key}

<div class="min-h-screen w-full flex justify-center dark:text-white">
	<div class=" py-2.5 flex flex-col justify-between w-full">
		<div class="max-w-2xl mx-auto w-full px-3 md:px-0 my-10">
			<div class="mb-6 flex justify-between items-center">
				<div class=" text-2xl font-semibold self-center">My Documents</div>
			</div>

			<div class=" flex w-full space-x-2">
				<div class="flex flex-1">
					<div class=" self-center ml-1 mr-3">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								fill-rule="evenodd"
								d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<input
						class=" w-full text-sm pr-4 py-1 rounded-r-xl outline-none bg-transparent"
						bind:value={query}
						placeholder="Search Document"
					/>
				</div>

				<div>
					<button
						class=" px-2 py-2 rounded-xl border border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 transition font-medium text-sm flex items-center space-x-1"
						on:click={() => {
							document.getElementById('upload-doc-input')?.click();
						}}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 16 16"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								d="M8.75 3.75a.75.75 0 0 0-1.5 0v3.5h-3.5a.75.75 0 0 0 0 1.5h3.5v3.5a.75.75 0 0 0 1.5 0v-3.5h3.5a.75.75 0 0 0 0-1.5h-3.5v-3.5Z"
							/>
						</svg>
					</button>
				</div>
			</div>

			<input
				id="upload-doc-input"
				bind:files={inputFiles}
				type="file"
				hidden
				on:change={async (e) => {
					if (inputFiles && inputFiles.length > 0) {
						const file = inputFiles[0];
						if (
							SUPPORTED_FILE_TYPE.includes(file['type']) ||
							SUPPORTED_FILE_EXTENSIONS.includes(file.name.split('.').at(-1))
						) {
							uploadDoc(file);
						} else {
							toast.error(
								`Unknown File Type '${file['type']}', but accepting and treating as plain text`
							);
							uploadDoc(file);
						}

						inputFiles = null;
						e.target.value = '';
					} else {
						toast.error(`File not found.`);
					}
				}}
			/>

			<div>
				<div
					class="my-3 py-16 rounded-lg border-2 border-dashed dark:border-gray-600 {dragged &&
						' dark:bg-gray-700'} "
					role="region"
					on:drop={onDrop}
					on:dragover={onDragOver}
					on:dragleave={onDragLeave}
				>
					<div class="  pointer-events-none">
						<div class="text-center dark:text-white text-2xl font-semibold z-50">Add Files</div>

						<div class=" mt-2 text-center text-sm dark:text-gray-200 w-full">
							Drop any files here to add to my documents
						</div>
					</div>
				</div>
			</div>

			{#each $documents.filter((p) => query === '' || p.name.includes(query)) as doc}
				<hr class=" dark:border-gray-700 my-2.5" />
				<div class=" flex space-x-4 cursor-pointer w-full mb-3">
					<div class=" flex flex-1 space-x-4 cursor-pointer w-full">
						<div class=" flex items-center space-x-3">
							<div class="p-2.5 bg-red-400 text-white rounded-lg">
								{#if doc}
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 24 24"
										fill="currentColor"
										class="w-6 h-6"
									>
										<path
											fill-rule="evenodd"
											d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0 0 16.5 9h-1.875a1.875 1.875 0 0 1-1.875-1.875V5.25A3.75 3.75 0 0 0 9 1.5H5.625ZM7.5 15a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5A.75.75 0 0 1 7.5 15Zm.75 2.25a.75.75 0 0 0 0 1.5H12a.75.75 0 0 0 0-1.5H8.25Z"
											clip-rule="evenodd"
										/>
										<path
											d="M12.971 1.816A5.23 5.23 0 0 1 14.25 5.25v1.875c0 .207.168.375.375.375H16.5a5.23 5.23 0 0 1 3.434 1.279 9.768 9.768 0 0 0-6.963-6.963Z"
										/>
									</svg>
								{:else}
									<svg
										class=" w-6 h-6 translate-y-[0.5px]"
										fill="currentColor"
										viewBox="0 0 24 24"
										xmlns="http://www.w3.org/2000/svg"
										><style>
											.spinner_qM83 {
												animation: spinner_8HQG 1.05s infinite;
											}
											.spinner_oXPr {
												animation-delay: 0.1s;
											}
											.spinner_ZTLf {
												animation-delay: 0.2s;
											}
											@keyframes spinner_8HQG {
												0%,
												57.14% {
													animation-timing-function: cubic-bezier(0.33, 0.66, 0.66, 1);
													transform: translate(0);
												}
												28.57% {
													animation-timing-function: cubic-bezier(0.33, 0, 0.66, 0.33);
													transform: translateY(-6px);
												}
												100% {
													transform: translate(0);
												}
											}
										</style><circle class="spinner_qM83" cx="4" cy="12" r="2.5" /><circle
											class="spinner_qM83 spinner_oXPr"
											cx="12"
											cy="12"
											r="2.5"
										/><circle class="spinner_qM83 spinner_ZTLf" cx="20" cy="12" r="2.5" /></svg
									>
								{/if}
							</div>
							<div class=" flex-1 self-center flex-1">
								<div class=" font-bold line-clamp-1">#{doc.name} ({doc.filename})</div>
								<div class=" text-xs overflow-hidden text-ellipsis line-clamp-1">
									{doc.title}
								</div>
							</div>
						</div>
					</div>
					<div class="flex flex-row space-x-1 self-center">
						<button
							class="self-center w-fit text-sm px-2 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							on:click={async () => {
								showEditDocModal = !showEditDocModal;
								selectedDoc = doc;
							}}
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="w-4 h-4"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487zm0 0L19.5 7.125"
								/>
							</svg>
						</button>

						<!-- <button
									class="self-center w-fit text-sm px-2 py-2 border dark:border-gray-600 rounded-xl"
									type="button"
									on:click={() => {
										console.log('download file');
									}}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 16 16"
										fill="currentColor"
										class="w-4 h-4"
									>
										<path
											d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.03 6.22a.75.75 0 0 0-1.06 1.06l3.5 3.5a.75.75 0 0 0 1.06 0l3.5-3.5a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75Z"
										/>
										<path
											d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z"
										/>
									</svg>
								</button> -->

						<button
							class="self-center w-fit text-sm px-2 py-2 border dark:border-gray-600 rounded-xl"
							type="button"
							on:click={() => {
								deleteDoc(doc.name);
							}}
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="w-4 h-4"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
								/>
							</svg>
						</button>
					</div>
				</div>
			{/each}
			{#if $documents.length != 0}
				<hr class=" dark:border-gray-700 my-2.5" />

				<div class=" flex justify-between w-full mb-3">
					<div class="flex space-x-2">
						<input
							id="documents-import-input"
							bind:files={importFiles}
							type="file"
							accept=".json"
							hidden
							on:change={() => {
								console.log(importFiles);

								const reader = new FileReader();
								reader.onload = async (event) => {
									const savedDocs = JSON.parse(event.target.result);
									console.log(savedDocs);

									for (const doc of savedDocs) {
										await createNewDoc(
											localStorage.token,
											doc.collection_name,
											doc.filename,
											doc.name,
											doc.title
										).catch((error) => {
											toast.error(error);
											return null;
										});
									}

									await documents.set(await getDocs(localStorage.token));
								};

								reader.readAsText(importFiles[0]);
							}}
						/>

						<button
							class="self-center w-fit text-sm px-3 py-1 border dark:border-gray-600 rounded-xl flex"
							on:click={async () => {
								document.getElementById('documents-import-input')?.click();
							}}
						>
							<div class=" self-center mr-2 font-medium">Import Documents Mapping</div>

							<div class=" self-center">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 16 16"
									fill="currentColor"
									class="w-4 h-4"
								>
									<path
										fill-rule="evenodd"
										d="M4 2a1.5 1.5 0 0 0-1.5 1.5v9A1.5 1.5 0 0 0 4 14h8a1.5 1.5 0 0 0 1.5-1.5V6.621a1.5 1.5 0 0 0-.44-1.06L9.94 2.439A1.5 1.5 0 0 0 8.878 2H4Zm4 9.5a.75.75 0 0 1-.75-.75V8.06l-.72.72a.75.75 0 0 1-1.06-1.06l2-2a.75.75 0 0 1 1.06 0l2 2a.75.75 0 1 1-1.06 1.06l-.72-.72v2.69a.75.75 0 0 1-.75.75Z"
										clip-rule="evenodd"
									/>
								</svg>
							</div>
						</button>

						<button
							class="self-center w-fit text-sm px-3 py-1 border dark:border-gray-600 rounded-xl flex"
							on:click={async () => {
								let blob = new Blob([JSON.stringify($documents)], {
									type: 'application/json'
								});
								saveAs(blob, `documents-mapping-export-${Date.now()}.json`);
							}}
						>
							<div class=" self-center mr-2 font-medium">Export Documents Mapping</div>

							<div class=" self-center">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 16 16"
									fill="currentColor"
									class="w-4 h-4"
								>
									<path
										fill-rule="evenodd"
										d="M4 2a1.5 1.5 0 0 0-1.5 1.5v9A1.5 1.5 0 0 0 4 14h8a1.5 1.5 0 0 0 1.5-1.5V6.621a1.5 1.5 0 0 0-.44-1.06L9.94 2.439A1.5 1.5 0 0 0 8.878 2H4Zm4 3.5a.75.75 0 0 1 .75.75v2.69l.72-.72a.75.75 0 1 1 1.06 1.06l-2 2a.75.75 0 0 1-1.06 0l-2-2a.75.75 0 0 1 1.06-1.06l.72.72V6.25A.75.75 0 0 1 8 5.5Z"
										clip-rule="evenodd"
									/>
								</svg>
							</div>
						</button>

						<!-- <button
						on:click={() => {
							loadDefaultPrompts();
						}}
					>
						dd
					</button> -->
					</div>
				</div>
			{/if}

			<div class="text-xs flex items-center space-x-1">
				<div>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						stroke-width="1.5"
						stroke="currentColor"
						class="w-3 h-3"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"
						/>
					</svg>
				</div>

				<div class="line-clamp-1">
					Tip: Use '#' in the prompt input to swiftly load and select your documents.
				</div>
			</div>
		</div>
	</div>
</div>
