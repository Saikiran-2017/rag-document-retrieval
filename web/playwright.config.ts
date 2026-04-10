import path from "node:path";
import { defineConfig } from "@playwright/test";

const webDir = __dirname;
const repoRoot = path.resolve(webDir, "..");
const e2eRaw = path.join(webDir, "tests-e2e", ".e2e-work", "raw");
const e2eFaiss = path.join(webDir, "tests-e2e", ".e2e-work", "faiss");

const apiPort = Number(process.env.E2E_API_PORT || 8000);
const webPort = Number(process.env.E2E_WEB_PORT || 3000);
const apiBase = `http://127.0.0.1:${apiPort}`;
const webBase = `http://127.0.0.1:${webPort}`;

const isWin = process.platform === "win32";
const sh = (cmd: string) => (isWin ? `cmd /c "${cmd}"` : cmd);

const apiProcessEnv = {
  ...process.env,
  PYTHONPATH: repoRoot,
  KA_RAW_DIR: e2eRaw,
  KA_FAISS_DIR: e2eFaiss,
  KA_DEBUG: "1",
};

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
  webServer: [
    {
      command: sh(
        `python -m uvicorn backend.app.main:app --host 127.0.0.1 --port ${apiPort}`,
      ),
      cwd: repoRoot,
      url: `${apiBase}/health`,
      // Avoid reusing an unknown/stale server on Windows; collision should fail fast.
      reuseExistingServer: false,
      timeout: 120_000,
      env: apiProcessEnv,
    },
    {
      // Avoid --turbopack here: SSE chat streaming is more reliable with the default webpack dev server.
      command: sh(`npx next dev -p ${webPort}`),
      cwd: webDir,
      url: webBase,
      reuseExistingServer: false,
      timeout: 180_000,
      env: {
        ...process.env,
        NEXT_PUBLIC_API_URL: apiBase,
        NEXT_PUBLIC_E2E_NO_STREAM: "1",
      },
    },
  ],
});
