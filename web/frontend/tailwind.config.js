/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        latin: ['"Crimson Pro"', 'serif'],
        ui: ['Inter', 'sans-serif'],
      },
      colors: {
        surface: {
          50: '#fafaf9',
          100: '#f5f5f4',
          200: '#e7e5e4',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        accent: {
          DEFAULT: '#6366f1',
          light: '#818cf8',
          dark: '#4f46e5',
        },
        highlight: {
          DEFAULT: '#f59e0b',
          light: '#fbbf24',
        },
        correct: {
          DEFAULT: '#10b981',
          light: '#34d399',
        },
        incorrect: {
          DEFAULT: '#f43f5e',
          light: '#fb7185',
        },
      },
      boxShadow: {
        glass: '0 4px 30px rgba(0, 0, 0, 0.06)',
        'glass-dark': '0 4px 30px rgba(0, 0, 0, 0.3)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 4px 1px rgba(99, 102, 241, 0.3)' },
          '50%': { boxShadow: '0 0 8px 3px rgba(99, 102, 241, 0.6)' },
        },
      },
    },
  },
  plugins: [],
}
