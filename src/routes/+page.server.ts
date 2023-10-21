import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => {
	const API_ENDPOINT = process.env.API_ENDPOINT;
	return {
		API_ENDPOINT
	};
};
