<script lang="ts">
	import TagInput from './Tags/TagInput.svelte';
	import TagList from './Tags/TagList.svelte';

	export let tags = [];

	export let deleteTag: Function;
	export let addTag: Function;
</script>

<div class="flex flex-row space-x-0.5 line-clamp-1">
	<TagList
		{tags}
		on:delete={(e) => {
			deleteTag(e.detail);
		}}
	/>

	<TagInput
		on:add={(e) => {
			addTag(e.detail);
		}}
	/>
</div>
