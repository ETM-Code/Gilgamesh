/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'spike': '#fbbf24',
        'excitatory': '#3b82f6',
        'inhibitory': '#ef4444',
      },
    },
  },
  plugins: [],
}
