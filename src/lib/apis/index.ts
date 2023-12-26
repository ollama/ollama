export const getOpenAIModels = async (
	base_url: string = 'https://api.openai.com/v1',
	api_key: string = ''
) => {
	let error = null;

	const res = await fetch(`${base_url}/models`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${api_key}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((error) => {
			console.log(error);
			error = `OpenAI: ${error?.error?.message ?? 'Network Problem'}`;
			return null;
		});

	if (error) {
		throw error;
	}

	let models = Array.isArray(res) ? res : res?.data ?? null;

	console.log(models);

	return models
		.map((model) => ({ name: model.id, external: true }))
		.filter((model) => (base_url.includes('openai') ? model.name.includes('gpt') : true));
};
