/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          navy: '#003366',
          accent: '#0066CC',
          lightblue: '#E6F0FA',
          panel: '#F8F9FB',
          border: '#E5E7EB',
        },
        risk: {
          high: '#D73027',
          moderate: '#FC8D59',
          low: '#FEE08B',
          minimal: '#1A9850',
        },
        tumor: {
          edema: '#FFD700',
          enhancing: '#EB3322',
          necrotic: '#3373D9',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};