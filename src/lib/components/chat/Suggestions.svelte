<script lang="ts">
	export let submitPrompt: Function;
</script>

<div class=" grid sm:grid-cols-2 gap-2.5 mb-4 md:p-2 text-left">
	<button
		class=" flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(`Tell me a random fun fact about the Roman Empire`);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium text-gray-300">Tell me a fun fact</div>
			<div class="text-sm text-gray-500">about the Roman Empire</div>
		</div>

		<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" class="w-4 h-4"
				><path
					d="M.5 1.163A1 1 0 0 1 1.97.28l12.868 6.837a1 1 0 0 1 0 1.766L1.969 15.72A1 1 0 0 1 .5 14.836V10.33a1 1 0 0 1 .816-.983L8.5 8 1.316 6.653A1 1 0 0 1 .5 5.67V1.163Z"
					fill="currentColor"
				/></svg
			>
		</div>
	</button>

	<button
		class="flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(`Show me a code snippet of a website's sticky header in CSS and JavaScript.`);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium text-gray-300">Show me a code snippet</div>
			<div class="text-sm text-gray-500">of a website's sticky header</div>
		</div>
		<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" class="w-4 h-4"
				><path
					d="M.5 1.163A1 1 0 0 1 1.97.28l12.868 6.837a1 1 0 0 1 0 1.766L1.969 15.72A1 1 0 0 1 .5 14.836V10.33a1 1 0 0 1 .816-.983L8.5 8 1.316 6.653A1 1 0 0 1 .5 5.67V1.163Z"
					fill="currentColor"
				/></svg
			>
		</div>
	</button>

	<button
		class=" hidden sm:flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(
				`Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option.`
			);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium text-gray-300">Help me study</div>
			<div class="text-sm text-gray-500">vocabulary for a college entrance exam</div>
		</div>
		<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" class="w-4 h-4"
				><path
					d="M.5 1.163A1 1 0 0 1 1.97.28l12.868 6.837a1 1 0 0 1 0 1.766L1.969 15.72A1 1 0 0 1 .5 14.836V10.33a1 1 0 0 1 .816-.983L8.5 8 1.316 6.653A1 1 0 0 1 .5 5.67V1.163Z"
					fill="currentColor"
				/></svg
			>
		</div>
	</button>

	<button
		class="  hidden sm:flex justify-between w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 outline outline-1 outline-gray-600 rounded-lg transition group"
		on:click={() => {
			submitPrompt(
				`What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.`
			);
		}}
	>
		<div class="flex flex-col text-left">
			<div class="text-sm font-medium text-gray-300">Give me ideas</div>
			<div class="text-sm text-gray-500">for what to do with my kids' art</div>
		</div>
		<div class="self-center group-hover:text-gray-300 text-gray-800 transition">
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" class="w-4 h-4"
				><path
					d="M.5 1.163A1 1 0 0 1 1.97.28l12.868 6.837a1 1 0 0 1 0 1.766L1.969 15.72A1 1 0 0 1 .5 14.836V10.33a1 1 0 0 1 .816-.983L8.5 8 1.316 6.653A1 1 0 0 1 .5 5.67V1.163Z"
					fill="currentColor"
				/></svg
			>
		</div>
	</button>
</div>
