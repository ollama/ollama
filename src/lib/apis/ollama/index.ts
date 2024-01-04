import { OLLAMA_API_BASE_URL } from '$lib/constants';

export const getOllamaVersion = async (
	base_url: string = OLLAMA_API_BASE_URL,
	token: string = ''
) => {
	let error = null;

	const res = await fetch(`${base_url}/version`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.log(err);
			if ('detail' in err) {
				error = err.detail;
			} else {
				error = 'Server connection failed';
			}
			return null;
		});

	if (error) {
		throw error;
	}

	return res?.version ?? '';
};

export const getOllamaModels = async (
	base_url: string = OLLAMA_API_BASE_URL,
	token: string = ''
) => {
	let error = null;

	const res = await fetch(`${base_url}/tags`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.log(err);
			if ('detail' in err) {
				error = err.detail;
			} else {
				error = 'Server connection failed';
			}
			return null;
		});

	if (error) {
		throw error;
	}

	return (res?.models ?? []).sort((a, b) => {
		return a.name.localeCompare(b.name);
	});
};

export const generateTitle = async (
	base_url: string = OLLAMA_API_BASE_URL,
	token: string = '',
	model: string,
	prompt: string
) => {
	let error = null;

	const res = await fetch(`${base_url}/generate`, {
		method: 'POST',
		headers: {
			'Content-Type': 'text/event-stream',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			model: model,
			prompt: `Generate a brief 3-5 word title for this question, excluding the term 'title.' Then, please reply with only the title: ${prompt}`,
			stream: false
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.log(err);
			if ('detail' in err) {
				error = err.detail;
			}
			return null;
		});

	if (error) {
		throw error;
	}

	return res?.response ?? 'New Chat';
};

export const generateChatCompletion = async (
	base_url: string = OLLAMA_API_BASE_URL,
	token: string = '',
	body: object
) => {
	let error = null;

	const res = await fetch(`${base_url}/chat`, {
		method: 'POST',
		headers: {
			'Content-Type': 'text/event-stream',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify(body)
	}).catch((err) => {
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return res;
};

export const createModel = async (
	base_url: string = OLLAMA_API_BASE_URL,
	token: string,
	tagName: string,
	content: string
) => {
	let error = null;

	const res = await fetch(`${base_url}/create`, {
		method: 'POST',
		headers: {
			'Content-Type': 'text/event-stream',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			name: tagName,
			modelfile: content
		})
	}).catch((err) => {
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return res;
};

export const deleteModel = async (
	base_url: string = OLLAMA_API_BASE_URL,
	token: string,
	tagName: string
) => {
	let error = null;

	const res = await fetch(`${base_url}/delete`, {
		method: 'DELETE',
		headers: {
			'Content-Type': 'text/event-stream',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			name: tagName
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			console.log(json);
			return true;
		})
		.catch((err) => {
			console.log(err);
			error = err.error;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
