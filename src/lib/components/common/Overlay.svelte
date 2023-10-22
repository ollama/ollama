<script>
	import Spinner from './Spinner.svelte';

	export let show = false;
	export let content = '';

	export let opacity = 1;
</script>

<div class="relative">
	{#if show}
		<div class="absolute w-full h-full flex">
			<div
				class="absolute rounded"
				style="inset: -10px; opacity: {opacity}; backdrop-filter: blur(5px);"
			/>

			<div class="flex w-full flex-col justify-center">
				<div class=" py-3">
					<Spinner className="ml-2" />
				</div>

				{#if content !== ''}
					<div class="text-center text-gray-100 text-xs font-medium z-50">
						{content}
					</div>
				{/if}
			</div>
		</div>
	{/if}

	<slot />
</div>
