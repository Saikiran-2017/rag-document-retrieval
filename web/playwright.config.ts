import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests-e2e",
  timeout: 60_000,
  retries: 0,
  use: {
    baseURL: process.env.NEXT_PUBLIC_API_URL ? "http://127.0.0.1:3000" : "http://127.0.0.1:3000",
    headless: true,
    viewport: { width: 1200, height: 800 },
  },
});

