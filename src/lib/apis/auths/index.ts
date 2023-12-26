import { WEBUI_API_BASE_URL } from '$lib/constants';

export const userSignIn = async (email: string, password: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/auths/signin`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			email: email,
			password: password
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((error) => {
			console.log(error);

			error = error.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const userSignUp = async (name: string, email: string, password: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/auths/signup`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			name: name,
			email: email,
			password: password
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((error) => {
			console.log(error);
			error = error.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
