<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';

	import { goto } from '$app/navigation';
	import { chatId } from '$lib/stores';

	export let title: string = 'Ollama Web UI';
</script>

<div
	class=" fixed top-0 flex flex-row justify-center bg-white/95 dark:bg-gray-800/90 dark:text-gray-200 backdrop-blur-xl w-full z-30"
>
	<div class="basis-full">
		<nav class="py-3" id="nav">
			<div class=" flex max-w-3xl mx-auto px-3">
				<div class="flex w-full max-w-full overflow-hidden text-ellipsis whitespace-nowrap">
					<div class="pr-2">
						<button
							class=" cursor-pointer p-1 flex dark:hover:bg-gray-700 rounded-lg transition"
							on:click={async () => {
								console.log('newChat');
								goto('/');
								await chatId.set(uuidv4());
							}}
						>
							<div class=" m-auto self-center">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 20 20"
									fill="currentColor"
									class="w-5 h-5"
								>
									<path
										d="M5.433 13.917l1.262-3.155A4 4 0 017.58 9.42l6.92-6.918a2.121 2.121 0 013 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 01-.65-.65z"
									/>
									<path
										d="M3.5 5.75c0-.69.56-1.25 1.25-1.25H10A.75.75 0 0010 3H4.75A2.75 2.75 0 002 5.75v9.5A2.75 2.75 0 004.75 18h9.5A2.75 2.75 0 0017 15.25V10a.75.75 0 00-1.5 0v5.25c0 .69-.56 1.25-1.25 1.25h-9.5c-.69 0-1.25-.56-1.25-1.25v-9.5z"
									/>
								</svg>
							</div>
						</button>
					</div>
					<div
						class=" flex-1 self-center font-medium overflow-hidden text-ellipsis whitespace-nowrap w-[80vw] pr-4"
					>
						{title != '' ? title : 'Ollama Web UI'}
					</div>
				</div>
			</div>
		</nav>
	</div>
</div>
