import { defineConfig } from "@playwright/test";

const webPort = Number(process.env.E2E_WEB_PORT || 3001);
const webBase = process.env.E2E_WEB_BASE || `http://127.0.0.1:${webPort}`;

export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 240_000,
  expect: { timeout: 45_000 },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: webBase,
    headless: true,
    viewport: { width: 1280, height: 900 },
    trace: "retain-on-failure",
    video: "off",
  },
});

