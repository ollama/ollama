/** @type {import('tailwindcss').Config} */
export default {
	darkMode: 'class',
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			colors: {
				gray: {
					50: '#f7f7f8',
					100: '#ececf1',
					200: '#d9d9e3',
					300: '#c5c5d2',
					400: '#acacbe',
					500: '#8e8ea0',
					600: '#565869',
					700: '#40414f',
					800: '#343541',
					900: '#202123',
					950: '#050509'
				}
			},
			typography: {
				DEFAULT: {
					css: {
						pre: false,
						code: false,
						'pre code': false,
						'code::before': false,
						'code::after': false
					}
				}
			}
		}
	},
	plugins: [require('@tailwindcss/typography')]
};
