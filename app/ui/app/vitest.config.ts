import { defineConfig, mergeConfig } from "vite";
import { configDefaults } from "vitest/config";
import path from "path";
import baseConfig from "./vite.config";

export default defineConfig((configEnv) =>
  mergeConfig(
    baseConfig(configEnv),
    defineConfig({
      resolve: {
        alias: {
          "@": path.resolve(__dirname, "./src"),
          "@/gotypes": path.resolve(__dirname, "./codegen/gotypes.gen.ts"),
        },
      },
      test: {
        environment: "node",
        globals: true,
        ...(configEnv.mode === "browser"
          ? {
              include: ["src/**/*.browser.test.tsx"],
              browser: {
                enabled: true,
                provider: "playwright",
                instances: [{ browser: "chromium" }],
                headless: true,
              },
            }
          : {
              exclude: [...configDefaults.exclude, "**/*.browser.test.tsx"],
            }),
      },
    }),
  ),
);
