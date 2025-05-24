// https://nuxt.com/docs/api/configuration/nuxt-config
import { definePreset } from "@primevue/themes";
import Aura from "@primevue/themes/aura";

const port = process.env.LOCAL_FRONT ? parseInt(process.env.LOCAL_FRONT) : 3000;
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
  ssr: false,
  devtools: { enabled: false },
  modules: [
    "@nuxtjs/tailwindcss",
    "@primevue/nuxt-module",
    "@pinia/nuxt",
    "@vueuse/nuxt",
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
  devServer: {
    port,
    host: "0.0.0.0",
  },
  runtimeConfig: {
    public: {
      WHERE: process.env.FRONT_WHERE,
    },
  },
});
