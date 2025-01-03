import vueTsEslintConfig from "@vue/eslint-config-typescript";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import pluginVue from "eslint-plugin-vue";
import globals from "globals";
import vueEslintParser from "vue-eslint-parser";
import prettierConfig from "@vue/eslint-config-prettier";

export default [
  ...pluginVue.configs["flat/essential"],
  ...vueTsEslintConfig({
    extends: ["recommended"],
  }),
  prettierConfig,
  {
    files: ["**/*.vue", "**/*.ts", "**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      parser: vueEslintParser,
      parserOptions: {
        ecmaVersion: "latest",
        parser: "@typescript-eslint/parser",
        sourceType: "module",
      },
      globals: {
        ...globals.browser,
        ...globals.node,
        myCustomGlobal: "readonly",
      },
    },
    plugins: {
      "simple-import-sort": simpleImportSort,
    },
    rules: {
      "vue/multi-word-component-names": "off",
      "simple-import-sort/imports": "error",
      "simple-import-sort/exports": "error",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_|(props)",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/no-explicit-any": "off",
      "vue/no-v-text-v-html-on-component": "off",
    },
  },
  {
    ignores: [
      "node_modules/**/*",
      "eslint.config.js",
      "prismicio-types.d.ts",
      "dist/**/*",
      ".*/**/*",
    ],
  },
];
