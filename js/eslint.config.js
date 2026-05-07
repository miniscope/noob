import js from "@eslint/js";
import globals from "globals";
import eslintConfigPrettier from "eslint-config-prettier";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import vitest from "@vitest/eslint-plugin";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommendedTypeChecked,
      tseslint.configs.stylistic,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        projectService: true,
      },
    },
    rules: {
      "no-unused-vars": "off",
      "no-console": "error",
      "@typescript-eslint/no-unused-vars": "error",
      "@typescript-eslint/consistent-indexed-object-style": "error",
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-empty-object-type": [
        "error",
        {
          allowInterfaces: "with-single-extends",
        },
      ],
    },
  },
  {
    files: ["tests/**"],
    plugins: {
      vitest,
    },
    rules: {
      ...vitest.configs.all.rules,
      "vitest/prefer-expect-assertions": "off",
      "vitest/consistent-test-it": "off",
      "vitest/require-top-level-describe": "off",
      "vitest/no-hooks": "off",
    },
    settings: {
      vitest: {
        typecheck: true,
      },
    },
    languageOptions: {
      globals: {
        ...vitest.environments.env.globals,
      },
    },
  },
  {
    // Config
    files: ["eslint.config.js"],
    ...tseslint.configs.disableTypeChecked,
  },
  eslintConfigPrettier,
]);
