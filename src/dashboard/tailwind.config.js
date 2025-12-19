/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: '#2563eb',
        danger: '#dc2626',
        warning: '#f59e0b',
        success: '#10b981',
      }
    }
  },
  plugins: [],
}
