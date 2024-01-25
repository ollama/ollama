import { writable } from 'svelte/store';

// Backend
export const config = writable(undefined);
export const user = writable(undefined);

// Frontend
export const theme = writable('dark');

export const chatId = writable('');

export const chats = writable([]);
export const tags = writable([]);
export const models = writable([]);
export const voices = writable([]);

export const modelfiles = writable([]);
export const prompts = writable([]);
export const documents = writable([
	{
		collection_name: 'collection_name',
		filename: 'filename',
		name: 'name',
		title: 'title'
	},
	{
		collection_name: 'collection_name1',
		filename: 'filename1',
		name: 'name1',
		title: 'title1'
	}
]);

export const settings = writable({});
export const showSettings = writable(false);
