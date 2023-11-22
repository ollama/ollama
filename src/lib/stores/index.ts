import { writable } from 'svelte/store';

// Backend
export const config = writable(undefined);
export const user = writable(undefined);

// Frontend
export const db = writable(undefined);
export const chatId = writable('');
export const chats = writable([]);
export const models = writable([]);
export const settings = writable({});
export const showSettings = writable(false);
