import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import postcssPresetEnv from "postcss-preset-env";
import { resolve } from "path";

export default defineConfig(() => ({
  base: "/",

  plugins: [
    TanStackRouterVite({ target: "react" }),
    react(),
    tailwindcss(),
    tsconfigPaths(),
  ],

  resolve: {
    alias: {
      "@/gotypes": resolve(__dirname, "codegen/gotypes.gen.ts"),
      "@": resolve(__dirname, "src"),
      "micromark-extension-math": "micromark-extension-llm-math",
    },
  },

  css: {
    postcss: {
      plugins: [
        postcssPresetEnv({
          stage: 1, // Include more experimental features that Safari 14 needs
          browsers: ["Safari >= 14"],
          // autoprefixer: false,
          features: {
            "custom-properties": true, // Let TailwindCSS handle this
            "nesting-rules": true,
            "logical-properties-and-values": true, // Polyfill logical properties
            "media-query-ranges": true, // Modern media query syntax
            "color-function": true, // CSS color functions
            "double-position-gradients": true,
            "gap-properties": true, // This is key for flexbox gap!
            "place-properties": true,
            "overflow-property": true,
            "focus-visible-pseudo-class": true, // Focus-visible support
            "focus-within-pseudo-class": true, // Focus-within support
            "any-link-pseudo-class": true, // :any-link pseudo-class
            "not-pseudo-class": true, // Enhanced :not() support
            "dir-pseudo-class": true, // :dir() pseudo-class
            "all-property": true, // CSS 'all' property
            "image-set-function": true, // image-set() function
            "hwb-function": true, // hwb() color function
            "lab-function": true, // lab() color function
            "oklab-function": true, // oklab() color function
          },
        }),
      ],
    },
  },

  build: {
    target: "es2017",
  },

  esbuild: {
    target: "es2017",
  },
}));
