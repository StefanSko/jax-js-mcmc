import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests
    },
  },
  test: {
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
      instances: [{ browser: "chromium" }],
    },
    testTimeout: 60000, // HMC tests can take time
    hookTimeout: 30000,
    setupFiles: ["tests/setup.ts"],
  },
});
