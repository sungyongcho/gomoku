/** @type {import('tailwindcss').Config} */

import screens from "./config/tailwind/screens.js";
import typo from "./config/tailwind/typo.js";

export default {
  content: [
    "./components/**/*.{js,vue,ts}",
    "./layouts/**/*.vue",
    "./pages/**/*.vue",
    "./content/**/*.md",
    "./plugins/**/*.{js,ts}",
    "./error.vue",
  ],
  theme: {
    extend: {
      boxShadow: {
        sm: "0 0 2px 0 rgb(0 0 0 / 0.05)",
        md: "0 0 6px -1px rgb(0 0 0 / 0.1), 0 0 4px -2px rgb(0 0 0 / 0.1)",
        lg: "0 0 15px -3px rgb(0 0 0 / 0.1), 0 0 6px -4px rgb(0 0 0 / 0.1)",
        xl: "0 0 25px -5px rgb(0 0 0 / 0.1), 0 0 10px -6px rgb(0 0 0 / 0.1)",
        "2xl": "0 0 50px -12px rgb(0 0 0 / 0.25)",
      },
    },
    screens,
  },
  plugins: [
    ({ addUtilities, matchUtilities }) => {
      // Add custom typo classes
      addUtilities({
        ...typo,
        ".hyphens": {
          hyphens: "auto",
          wordBreak: "break-word",
        },
      });
      // Add path fill classes
      matchUtilities({
        path: (value) => ({
          path: {
            fill: value,
          },
        }),
      });
    },
  ],
};
