import { ENDPOINT } from '$lib/constants';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url }) => {
	const OLLAMA_ENDPOINT = process.env.OLLAMA_ENDPOINT;
	console.log(OLLAMA_ENDPOINT);
	const models = await fetch(
		`${OLLAMA_ENDPOINT != undefined ? OLLAMA_ENDPOINT : ENDPOINT}/api/tags`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json'
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((error) => {
			console.log(error);
			return null;
		});

	return {
		models: models?.models ?? [],
		OLLAMA_ENDPOINT: process.env.OLLAMA_ENDPOINT
	};
};
