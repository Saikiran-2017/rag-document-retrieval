import path from "node:path";
import { defineConfig } from "@playwright/test";

const webDir = __dirname;
const repoRoot = path.resolve(webDir, "..");
const e2eRaw = path.join(webDir, "tests-e2e", ".e2e-work", "raw");
const e2eFaiss = path.join(webDir, "tests-e2e", ".e2e-work", "faiss");

const apiProcessEnv = {
  ...process.env,
  PYTHONPATH: repoRoot,
  KA_RAW_DIR: e2eRaw,
  KA_FAISS_DIR: e2eFaiss,
};

export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 240_000,
  expect: { timeout: 45_000 },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: "http://127.0.0.1:3000",
    headless: true,
    viewport: { width: 1280, height: 900 },
    trace: "retain-on-failure",
    video: "off",
  },
  webServer: [
    {
      command: "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000",
      cwd: repoRoot,
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: apiProcessEnv,
    },
    {
      // Avoid --turbopack here: SSE chat streaming is more reliable with the default webpack dev server.
      command: "npx next dev -p 3000",
      cwd: webDir,
      url: "http://127.0.0.1:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        ...process.env,
        NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000",
      },
    },
  ],
});
