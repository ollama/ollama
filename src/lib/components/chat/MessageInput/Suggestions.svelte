<script lang="ts">
	export let submitPrompt: Function;
</script>

<div class=" grid sm:grid-cols-2 gap-2.5 mb-4 md:p-2 text-left">
	<button
		class=" flex justify-between w-full px-4 py-2.5 bg-white hover:bg-gray-50 dark:bg-gray-800 dark:hover:bg-gray-700 outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(`Tell me a random fun fact about the Roman Empire`);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium dark:text-gray-300">Tell me a fun fact</div>
			<div class="text-sm text-gray-500">about the Roman Empire</div>
		</div>

		<div
			class="self-center p-1 rounded-lg text-white group-hover:bg-gray-100 group-hover:text-gray-800 dark:group-hover:bg-gray-800 dark:group-hover:text-gray-300 dark:text-gray-800 transition"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 20 20"
				fill="currentColor"
				class="w-4 h-4"
			>
				<path
					fill-rule="evenodd"
					d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
					clip-rule="evenodd"
				/>
			</svg>
		</div>
	</button>

	<button
		class=" flex justify-between w-full px-4 py-2.5 bg-white hover:bg-gray-50 dark:bg-gray-800 dark:hover:bg-gray-700 outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(`Show me a code snippet of a website's sticky header in CSS and JavaScript.`);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium dark:text-gray-300">Show me a code snippet</div>
			<div class="text-sm text-gray-500">of a website's sticky header</div>
		</div>
		<div
			class="self-center p-1 rounded-lg text-white group-hover:bg-gray-100 group-hover:text-gray-800 dark:group-hover:bg-gray-800 dark:group-hover:text-gray-300 dark:text-gray-800 transition"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 20 20"
				fill="currentColor"
				class="w-4 h-4"
			>
				<path
					fill-rule="evenodd"
					d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
					clip-rule="evenodd"
				/>
			</svg>
		</div>
	</button>

	<button
		class="  hidden sm:flex justify-between w-full px-4 py-2.5 bg-white hover:bg-gray-50 dark:bg-gray-800 dark:hover:bg-gray-700 outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(
				`Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option.`
			);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium dark:text-gray-300">Help me study</div>
			<div class="text-sm text-gray-500">vocabulary for a college entrance exam</div>
		</div>
		<div
			class="self-center p-1 rounded-lg text-white group-hover:bg-gray-100 group-hover:text-gray-800 dark:group-hover:bg-gray-800 dark:group-hover:text-gray-300 dark:text-gray-800 transition"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 20 20"
				fill="currentColor"
				class="w-4 h-4"
			>
				<path
					fill-rule="evenodd"
					d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
					clip-rule="evenodd"
				/>
			</svg>
		</div>
	</button>

	<button
		class="  hidden sm:flex justify-between w-full px-4 py-2.5 bg-white hover:bg-gray-50 dark:bg-gray-800 dark:hover:bg-gray-700 outline outline-1 outline-gray-200 dark:outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(
				`What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.`
			);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium dark:text-gray-300">Give me ideas</div>
			<div class="text-sm text-gray-500">for what to do with my kids' art</div>
		</div>
		<div
			class="self-center p-1 rounded-lg text-white group-hover:bg-gray-100 group-hover:text-gray-800 dark:group-hover:bg-gray-800 dark:group-hover:text-gray-300 dark:text-gray-800 transition"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 20 20"
				fill="currentColor"
				class="w-4 h-4"
			>
				<path
					fill-rule="evenodd"
					d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z"
					clip-rule="evenodd"
				/>
			</svg>
		</div>
	</button>
</div>
