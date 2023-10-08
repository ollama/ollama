<script>
	import Modal from '../common/Modal.svelte';
	export let show = false;
</script>

<Modal bind:show>
	<div class="mt-3 p-3 rounded-lg bg-gray-900">
		<label for="models" class="block mb-2 text-sm font-medium text-gray-200">Select a model</label>
		<select
			id="models"
			class="border border-gray-600 bg-gray-700 text-gray-200 text-sm rounded-lg block w-full p-2.5 placeholder-gray-400"
		>
			<option value="US">United States</option>
			<option value="CA">Canada</option>
			<option value="FR">France</option>
			<option value="DE">Germany</option>
		</select>
	</div>
</Modal>
