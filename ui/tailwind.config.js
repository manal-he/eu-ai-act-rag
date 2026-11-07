/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["'Source Serif 4'", "Georgia", "serif"],
      },
      colors: {
        eu: { 50: "#eef3fb", 100: "#d6e2f5", 500: "#2a5ab8", 600: "#1f4796", 700: "#183872", 900: "#0d1f40" },
        star: "#f2c230",
      },
    },
  },
  plugins: [],
};
