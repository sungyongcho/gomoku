// https://nuxt.com/docs/api/configuration/nuxt-config
import { definePreset } from "@primevue/themes";
import Aura from "@primevue/themes/aura";

const port = process.env.LOCAL_FRONT ? parseInt(process.env.LOCAL_FRONT) : 3000;
const LOCAL_FRONT_NUXT_CONTENT_WS = parseInt(process.env.LOCAL_FRONT_NUXT_CONTENT_WS || '4000');

const Noir = definePreset(Aura, {
  semantic: {
    primary: {
      50: "{zinc.50}",
      100: "{zinc.100}",
      200: "{zinc.200}",
      300: "{zinc.300}",
      400: "{zinc.400}",
      500: "{zinc.500}",
      600: "{zinc.600}",
      700: "{zinc.700}",
      800: "{zinc.800}",
      900: "{zinc.900}",
      950: "{zinc.950}",
    },
    colorScheme: {
      light: {
        primary: {
          color: "{zinc.950}",
          inverseColor: "#ffffff",
          hoverColor: "{zinc.900}",
          activeColor: "{zinc.800}",
        },
        highlight: {
          background: "{zinc.950}",
          focusBackground: "{zinc.700}",
          color: "#ffffff",
          focusColor: "#ffffff",
        },
      },
      dark: {
        primary: {
          color: "{zinc.50}",
          inverseColor: "{zinc.950}",
          hoverColor: "{zinc.100}",
          activeColor: "{zinc.200}",
        },
        highlight: {
          background: "rgba(250, 250, 250, .16)",
          focusBackground: "rgba(250, 250, 250, .24)",
          color: "rgba(255,255,255,.87)",
          focusColor: "rgba(255,255,255,.87)",
        },
      },
    },
  },
});

export default defineNuxtConfig({
  compatibilityDate: "2025-01-09",
  ssr: true,
  devtools: { enabled: false },
  modules: [
    "@nuxtjs/tailwindcss",
    "@primevue/nuxt-module",
    "@pinia/nuxt",
    "@vueuse/nuxt",
    "@nuxt/content",
  ],
  primevue: {
    options: {
      ripple: true,
      theme: {
        preset: Noir,
        options: {
          darkModeSelector: ".dark",
        },
      },
    },
  },
  components: [
    {
      global: true,
      path: "~/components",
      pathPrefix: false,
    },
  ],
  css: ["~/assets/styles/main.css", "~/assets/styles/main.scss"],
  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },
  pinia: {
    storesDirs: ["./stores/**"],
  },
  content: {
    watch: {
      enabled: true,
      port: LOCAL_FRONT_NUXT_CONTENT_WS,
      showURL: true,
    },
    build: {
      markdown: {
        highlight: {
          theme: "github-dark",
          preload: [
            "js",
            "vue",
            "html",
            "css",
            "json",
            "ts",
            "python",
            "solidity",
          ],
        },
      },
    },
  },
  devServer: {
    port,
  },
  runtimeConfig: {
    public: {
      FRONT_WHERE: process.env.FRONT_WHERE || "local",
      LOCAL_MINIMAX: process.env.LOCAL_MINIMAX,
      LOCAL_ALPHAZERO: process.env.LOCAL_ALPHAZERO,
      LOCAL_FRONT_NUXT_CONTENT_WS: LOCAL_FRONT_NUXT_CONTENT_WS,
    },
  },
});
