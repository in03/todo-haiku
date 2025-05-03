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
      },
      fontFamily: {
        serif: ['Noto Serif', 'serif'],
      },
      boxShadow: {
        task: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        'task-hover': '0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23)',
      },
      minHeight: {
        'kanban-column': '500px', // Increase the minimum height for kanban columns
      },
      width: {
        'kanban-board': '100%', // Full width for the kanban board
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    // Allows prefixing tailwind classes with LiveView classes to add rules
    // only when LiveView classes are applied, for example:
    //
    //     <div class="phx-click-loading:animate-ping">
    //
    plugin(({addVariant}) => addVariant("phx-click-loading", [".phx-click-loading&", ".phx-click-loading &"])),
    plugin(({addVariant}) => addVariant("phx-submit-loading", [".phx-submit-loading&", ".phx-submit-loading &"])),
    plugin(({addVariant}) => addVariant("phx-change-loading", [".phx-change-loading&", ".phx-change-loading &"])),

    // Embeds Heroicons (https://heroicons.com) into your app.css bundle
    // See your `CoreComponents.icon/1` for more information.
    //
    plugin(function({matchComponents, theme}) {
      let iconsDir = path.join(__dirname, "../deps/heroicons/optimized")
      let values = {}
      let icons = [
        ["", "/24/outline"],
        ["-solid", "/24/solid"],
        ["-mini", "/20/solid"],
        ["-micro", "/16/solid"]
      ]
      icons.forEach(([suffix, dir]) => {
        fs.readdirSync(path.join(iconsDir, dir)).forEach(file => {
          let name = path.basename(file, ".svg") + suffix
          values[name] = {name, fullPath: path.join(iconsDir, dir, file)}
        })
      })
      matchComponents({
        "hero": ({name, fullPath}) => {
          let content = fs.readFileSync(fullPath).toString().replace(/\r?\n|\r/g, "")
          let size = theme("spacing.6")
          if (name.endsWith("-mini")) {
            size = theme("spacing.5")
          } else if (name.endsWith("-micro")) {
            size = theme("spacing.4")
          }
          return {
            [`--hero-${name}`]: `url('data:image/svg+xml;utf8,${content}')`,
            "-webkit-mask": `var(--hero-${name})`,
            "mask": `var(--hero-${name})`,
            "mask-repeat": "no-repeat",
            "background-color": "currentColor",
            "vertical-align": "middle",
            "display": "inline-block",
            "width": size,
            "height": size
          }
        }
      }, {values})
    })
  ]
}
