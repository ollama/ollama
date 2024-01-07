import { RAG_API_BASE_URL } from '$lib/constants';

export const uploadDocToVectorDB = async (token: string, collection_name: string, file: File) => {
	const data = new FormData();
	data.append('file', file);
	data.append('collection_name', collection_name);

	let error = null;

	const res = await fetch(`${RAG_API_BASE_URL}/doc`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		},
		body: data
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.log(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const uploadWebToVectorDB = async (token: string, collection_name: string, url: string) => {
	let error = null;

	const res = await fetch(`${RAG_API_BASE_URL}/web`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			url: url,
			collection_name: collection_name
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.log(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const queryVectorDB = async (
	token: string,
	collection_name: string,
	query: string,
	k: number
) => {
	let error = null;
	const searchParams = new URLSearchParams();

	searchParams.set('query', query);
	if (k) {
		searchParams.set('k', k.toString());
	}

	const res = await fetch(
		`${RAG_API_BASE_URL}/query/${collection_name}/?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const resetVectorDB = async (token: string) => {
	let error = null;

	const res = await fetch(`${RAG_API_BASE_URL}/reset`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
