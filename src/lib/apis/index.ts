import { WEBUI_API_BASE_URL } from '$lib/constants';

export const getBackendConfig = async () => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.log(err);
			error = err;
			return null;
		});

	return res;
};
