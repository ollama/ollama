<script lang="ts">
	import { copyToClipboard } from '$lib/utils';
	import hljs from 'highlight.js';
	import 'highlight.js/styles/github-dark.min.css';

	export let lang = '';
	export let code = '';

	let copied = false;

	const copyCode = async () => {
		copied = true;
		await copyToClipboard(code);

		setTimeout(() => {
			copied = false;
		}, 1000);
	};

	$: highlightedCode = code ? hljs.highlightAuto(code, hljs.getLanguage(lang)?.aliases).value : '';
</script>

{#if code}
	<div class="mb-4">
		<div
			class="flex justify-between bg-[#202123] text-white text-xs px-4 pt-1 pb-0.5 rounded-t-lg overflow-x-auto"
		>
			<div class="p-1">{@html lang}</div>
			<button class="copy-code-button bg-none border-none p-1" on:click={copyCode}
				>{copied ? 'Copied' : 'Copy Code'}</button
			>
		</div>

		<pre class=" rounded-b-lg hljs p-4 px-5 overflow-x-auto rounded-t-none"><code
				class="language-{lang} rounded-t-none whitespace-pre">{@html highlightedCode || code}</code
			></pre>
	</div>
{/if}
