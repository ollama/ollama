<script lang="ts">
	import { settings } from '$lib/stores';
	import Suggestions from './MessageInput/Suggestions.svelte';

	export let submitPrompt: Function;
	export let stopResponse: Function;

	export let suggestions = 'true';
	export let autoScroll = true;

	export let fileUploadEnabled = false;
	export let speechRecognitionEnabled = true;
	export let speechRecognitionListening = false;

	export let prompt = '';
	export let messages = [];

	let speechRecognition;

	const speechRecognitionHandler = () => {
		// Check if SpeechRecognition is supported

		if (speechRecognitionListening) {
			speechRecognition.stop();
		} else {
			if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
				// Create a SpeechRecognition object
				speechRecognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();

				// Set continuous to true for continuous recognition
				speechRecognition.continuous = true;

				// Set the timeout for turning off the recognition after inactivity (in milliseconds)
				const inactivityTimeout = 3000; // 3 seconds

				let timeoutId;
				// Start recognition
				speechRecognition.start();
				speechRecognitionListening = true;

				// Event triggered when speech is recognized
				speechRecognition.onresult = function (event) {
					// Clear the inactivity timeout
					clearTimeout(timeoutId);

					// Handle recognized speech
					console.log(event);
					const transcript = event.results[Object.keys(event.results).length - 1][0].transcript;
					prompt = `${prompt}${transcript}`;

					// Restart the inactivity timeout
					timeoutId = setTimeout(() => {
						console.log('Speech recognition turned off due to inactivity.');
						speechRecognition.stop();
					}, inactivityTimeout);
				};

				// Event triggered when recognition is ended
				speechRecognition.onend = function () {
					// Restart recognition after it ends
					console.log('recognition ended');
					speechRecognitionListening = false;
					if (prompt !== '' && $settings?.speechAutoSend === true) {
						submitPrompt(prompt);
					}
				};

				// Event triggered when an error occurs
				speechRecognition.onerror = function (event) {
					console.log(event);
					toast.error(`Speech recognition error: ${event.error}`);
					speechRecognitionListening = false;
				};
			} else {
				toast.error('SpeechRecognition API is not supported in this browser.');
			}
		}
	};
</script>

<div class="fixed bottom-0 w-full">
	<div class="  pt-5">
		<div class="max-w-3xl px-2.5 pt-2.5 -mb-0.5 mx-auto inset-x-0">
			{#if messages.length == 0 && suggestions !== 'false'}
				<Suggestions {submitPrompt} />
			{/if}

			{#if autoScroll === false && messages.length > 0}
				<div class=" flex justify-center mb-4">
					<button
						class=" bg-white border border-gray-100 dark:border-none dark:bg-white/20 p-1.5 rounded-full"
						on:click={() => {
							autoScroll = true;
							window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
						}}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-5 h-5"
						>
							<path
								fill-rule="evenodd"
								d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z"
								clip-rule="evenodd"
							/>
						</svg>
					</button>
				</div>
			{/if}

			<div class="bg-gradient-to-t from-white dark:from-gray-800 from-40% pb-2">
				<form
					class=" flex relative w-full"
					on:submit|preventDefault={() => {
						submitPrompt(prompt);
					}}
				>
					<textarea
						id="chat-textarea"
						class="rounded-xl dark:bg-gray-800 dark:text-gray-100 outline-none border dark:border-gray-600 w-full py-3
                        {fileUploadEnabled ? 'pl-12' : 'pl-5'} {speechRecognitionEnabled
							? 'pr-20'
							: 'pr-12'} resize-none"
						placeholder={speechRecognitionListening ? 'Listening...' : 'Send a message'}
						bind:value={prompt}
						on:keypress={(e) => {
							if (e.keyCode == 13 && !e.shiftKey) {
								e.preventDefault();
							}
							if (prompt !== '' && e.keyCode == 13 && !e.shiftKey) {
								submitPrompt(prompt);
							}
						}}
						rows="1"
						on:input={(e) => {
							e.target.style.height = '';
							e.target.style.height = Math.min(e.target.scrollHeight, 200) + 2 + 'px';
						}}
					/>

					{#if fileUploadEnabled}
						<div class=" absolute left-0 bottom-0">
							<div class="pl-2.5 pb-[9px]">
								<button
									class="  text-gray-600 dark:text-gray-200 transition rounded-lg p-1.5"
									type="button"
									on:click={() => {
										console.log('file');
									}}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 20 20"
										fill="currentColor"
										class="w-5 h-5"
									>
										<path
											fill-rule="evenodd"
											d="M15.621 4.379a3 3 0 00-4.242 0l-7 7a3 3 0 004.241 4.243h.001l.497-.5a.75.75 0 011.064 1.057l-.498.501-.002.002a4.5 4.5 0 01-6.364-6.364l7-7a4.5 4.5 0 016.368 6.36l-3.455 3.553A2.625 2.625 0 119.52 9.52l3.45-3.451a.75.75 0 111.061 1.06l-3.45 3.451a1.125 1.125 0 001.587 1.595l3.454-3.553a3 3 0 000-4.242z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							</div>
						</div>
					{/if}

					<div class=" absolute right-0 bottom-0">
						<div class="pr-2.5 pb-[9px]">
							{#if messages.length == 0 || messages.at(-1).done == true}
								{#if speechRecognitionEnabled}
									<button
										class=" text-gray-600 dark:text-gray-300 transition rounded-lg p-1 mr-0.5"
										type="button"
										on:click={() => {
											speechRecognitionHandler();
										}}
									>
										{#if speechRecognitionListening}
											<svg
												class=" w-5 h-5 translate-y-[0.5px]"
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
										{:else}
											<svg
												xmlns="http://www.w3.org/2000/svg"
												viewBox="0 0 20 20"
												fill="currentColor"
												class="w-5 h-5 translate-y-[0.5px]"
											>
												<path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
												<path
													d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z"
												/>
											</svg>
										{/if}
									</button>
								{/if}
								<button
									class="{prompt !== ''
										? 'bg-black text-white hover:bg-gray-900 dark:bg-white dark:text-black dark:hover:bg-gray-100 '
										: 'text-white bg-gray-100 dark:text-gray-800 dark:bg-gray-600 disabled'} transition rounded-lg p-1"
									type="submit"
									disabled={prompt === ''}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 20 20"
										fill="currentColor"
										class="w-5 h-5"
									>
										<path
											fill-rule="evenodd"
											d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							{:else}
								<button
									class="bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-800 transition rounded-lg p-1.5"
									on:click={stopResponse}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										viewBox="0 0 24 24"
										fill="currentColor"
										class="w-5 h-5"
									>
										<path
											fill-rule="evenodd"
											d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm6-2.438c0-.724.588-1.312 1.313-1.312h4.874c.725 0 1.313.588 1.313 1.313v4.874c0 .725-.588 1.313-1.313 1.313H9.564a1.312 1.312 0 01-1.313-1.313V9.564z"
											clip-rule="evenodd"
										/>
									</svg>
								</button>
							{/if}
						</div>
					</div>
				</form>

				<div class="mt-1.5 text-xs text-gray-500 text-center">
					LLMs can make mistakes. Verify important information.
				</div>
			</div>
		</div>
	</div>
</div>
