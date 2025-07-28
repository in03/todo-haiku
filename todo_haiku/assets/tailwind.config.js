// See the Tailwind configuration guide for advanced usage
// https://tailwindcss.com/docs/configuration

const plugin = require("tailwindcss/plugin")
const fs = require("fs")
const path = require("path")

module.exports = {
  content: [
    "./js/**/*.js",
    "../lib/*_web.ex",
    "../lib/*_web/**/*.*ex"
  ],
  theme: {
    extend: {
      colors: {
        brand: "#9d8cff", // Custom purple
        accent: {
          DEFAULT: "#9d8cff", // Custom purple
          foreground: "#FFFFFF",
        },
        muted: {
          DEFAULT: "#2a2a2a", // Dark gray (for dark mode)
          foreground: "#a0a0a0", // Light gray for text on dark
        },
        foreground: "#f2f2f2", // Light gray for text
        border: "#3a3a3a", // Darker gray for borders
        background: "#1a1a1a", // Very dark gray
        card: "#222222", // Slightly lighter than background
        destructive: {
          DEFAULT: "#ff6b6b", // Soft red
          foreground: "#FFFFFF",
        },
        // Neon colors
        neon: {
          blue: "#3b82f6",
          green: "#22c55e",
          yellow: "#eab308",
          red: "#ef4444",
          purple: "#9d8cff",
        },
      },
      fontFamily: {
        serif: ['Noto Serif', 'serif'],
      },
      boxShadow: {
        task: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        'task-hover': '0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23)',
        'neon-sm': '0 0 5px var(--tw-shadow-color, rgba(157, 140, 255, 0.7))',
        'neon-md': '0 0 10px var(--tw-shadow-color, rgba(157, 140, 255, 0.7))',
        'neon-lg': '0 0 15px var(--tw-shadow-color, rgba(157, 140, 255, 0.7))',
      },
      minHeight: {
        'kanban-column': '500px', // Increase the minimum height for kanban columns
      },
      width: {
        'kanban-board': '100%', // Full width for the kanban board
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(157, 140, 255, 0.7)' },
          '100%': { boxShadow: '0 0 15px rgba(157, 140, 255, 0.7)' },
        },
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms")
  ]
}
