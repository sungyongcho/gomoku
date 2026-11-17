import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Unit tests for the plain game-rule composables. They run outside Nuxt, so the
// `~` alias is set here and Nuxt auto-imports are stubbed in each test.
export default defineConfig({
  resolve: {
    alias: { "~": fileURLToPath(new URL("./", import.meta.url)) },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
