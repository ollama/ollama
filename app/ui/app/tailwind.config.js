/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      spacing: {
        3.5: "0.875rem",
        4.5: "1.125rem",
      },
      colors: {
        gray: {
          350: "#a1a1aa",
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
