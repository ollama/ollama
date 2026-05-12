import { defineConfig, mergeConfig } from "vite";
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
      },
    }),
  ),
);
