/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces
        surface: {
          page:         '#FAFAFA',
          card:         '#FFFFFF',
          'card-hover': '#F8F9FB',
          elevated:     '#FFFFFF',
          muted:        '#F1F3F5',
        },
        // Borders / lines (utility usage: border-border-subtle, etc.)
        border: {
          subtle:  '#E5E7EB',
          default: '#D1D5DB',
          strong:  '#9CA3AF',
        },
        // Text
        text: {
          primary:   '#111827',
          secondary: '#4B5563',
          tertiary:  '#6B7280',
          muted:     '#9CA3AF',
        },
        // Brand — clinical navy/blue family
        brand: {
          50:   '#EFF6FF',
          100:  '#DBEAFE',
          200:  '#BFDBFE',
          300:  '#93C5FD',
          500:  '#3B82F6',
          600:  '#2563EB',
          700:  '#1D4ED8',
          900:  '#1E3A8A',
          navy: '#003366',
        },
        // Semantic — eloquent-cortex risk colors
        risk: {
          high:     '#DC2626',
          moderate: '#F59E0B',
          low:      '#FBBF24',
          minimal:  '#10B981',
        },
        // Tumor class colors (sync with PlotlyBrainViewer + AnatomyCard bars)
        tumor: {
          edema:     '#FBBF24',
          enhancing: '#DC2626',
          necrotic:  '#2563EB',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'Consolas', 'monospace'],
      },
      borderRadius: {
        card: '12px',
      },
      boxShadow: {
        sm: '0 1px 2px rgba(15, 23, 42, 0.04)',
        md: '0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04)',
        lg: '0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04)',
        xl: '0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.04)',
      },
      transitionDuration: {
        DEFAULT: '200ms',
      },
    },
  },
  plugins: [],
};
