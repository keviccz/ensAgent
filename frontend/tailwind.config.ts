import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Core surfaces
        'surface-0': '#FFFFFF',
        'surface-1': '#F7F7F7',
        'surface-2': '#F0F0F0',
        'surface-3': '#E8E8E8',
        // Text
        'ink-heavy': '#0A0A0A',
        'ink-mid': '#3A3A3A',
        'ink-muted': '#7A7A7A',
        'ink-ghost': '#B0B0B0',
        // Controls
        'ctrl-dark': '#1A1A1A',
        'ctrl-hover': '#2A2A2A',
        // User bubble
        'bubble-user': '#DEDEDE',
        // Data / accent
        'data': '#0EA5E9',
        'data-dim': 'rgba(14,165,233,0.12)',
        'data-glow': 'rgba(14,165,233,0.25)',
        // Status
        'ok': '#22C55E',
        'warn': '#F59E0B',
        'err': '#EF4444',
        // Border
        'rule': 'rgba(0,0,0,0.07)',
        'rule-mid': 'rgba(0,0,0,0.12)',
      },
      fontFamily: {
        display: ['Sora', 'sans-serif'],
        body: ['Sora', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      fontSize: {
        '2xs': ['10px', { lineHeight: '14px', letterSpacing: '0.04em' }],
        'xs':  ['11px', { lineHeight: '16px' }],
        'sm':  ['12.5px', { lineHeight: '18px' }],
        'base':['14px',   { lineHeight: '20px' }],
        'lg':  ['16px',   { lineHeight: '24px' }],
        'xl':  ['20px',   { lineHeight: '28px' }],
        '2xl': ['24px',   { lineHeight: '32px' }],
        '3xl': ['30px',   { lineHeight: '36px' }],
      },
      borderRadius: {
        'sm': '6px',
        'DEFAULT': '8px',
        'md': '10px',
        'lg': '14px',
        'xl': '18px',
        '2xl': '24px',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'panel': '0 4px 16px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04)',
        'data': '0 0 12px rgba(14,165,233,0.2)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.4', transform: 'scale(0.75)' },
        },
        'cursor-blink': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        'slide-in': {
          '0%': { transform: 'translateX(-6px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.3s ease-out',
        'fade-in': 'fade-in 0.2s ease-out',
        'pulse-dot': 'pulse-dot 1.5s ease-in-out infinite',
        'cursor-blink': 'cursor-blink 1s step-end infinite',
        'slide-in': 'slide-in 0.2s ease-out',
        'shimmer': 'shimmer 2s linear infinite',
      },
    },
  },
  plugins: [],
}

export default config
