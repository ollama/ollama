<script>
	import { Toaster } from 'svelte-french-toast';

	import '../app.css';
	import '../tailwind.css';
</script>

<svelte:head>
	<title>Ollama</title>
</svelte:head>

<slot />
<Toaster />
