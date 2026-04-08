/**
 * Brutal E2E: UI smoke, API/upload edge cases, and grounded chat (OpenAI) when configured.
 *
 * Servers: playwright.config starts FastAPI (isolated KA_RAW_DIR / KA_FAISS_DIR) + Next dev.
 * LLM-heavy tests skip automatically when /health reports a missing or placeholder API key.
 */
import fs from "node:fs";
import path from "node:path";
import { test, expect } from "@playwright/test";
import {
  clearE2ELibrary,
  fetchHealth,
  postChatJson,
  shouldRunLlmSuite,
} from "./helpers/api";
import {
  clickSyncDocuments,
  openSidebarIfMobile,
  sendChatMessage,
  uploadCorpus,
  waitForAssistantAnyText,
  waitForLibraryFile,
  waitForSidebarLibraryReady,
} from "./helpers/ui";

const CORPUS_FILE = path.join(__dirname, "fixtures", "e2e_brutal_corpus.txt");
const CORPUS_BUFFER = fs.readFileSync(CORPUS_FILE);
const CORPUS_NAME = "e2e_brutal_corpus.txt";
const TOKEN = "E2E_VERIFY_TOKEN_XY42NZ";
const CEO_ANCHOR = "BRUTAL_E2E_CEO_ANCHOR";

test.describe("A. Smoke — shell loads", () => {
  test("home shows product chrome and library actions", async ({ page }) => {
    await page.goto("/");
    // Desktop: title lives in #app-sidebar. Mobile: duplicate in header is visibility:hidden on lg.
    await expect(page.locator("#app-sidebar").getByText("Knowledge Assistant")).toBeVisible();
    await openSidebarIfMobile(page);
    await expect(page.getByRole("button", { name: /Upload documents/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Sync documents/i })).toBeVisible();
    await expect(page.getByPlaceholder("Message…")).toBeVisible();
  });

  test("health endpoint reports API up", async ({ request }) => {
    const h = await fetchHealth(request);
    expect(h).not.toBeNull();
    expect(h?.status).toBe("ok");
  });
});

test.describe("B. API flows — uploads & sync UX (no chat model)", () => {
  test.describe.configure({ mode: "serial" });

  test.beforeEach(async ({ request }) => {
    const h = await fetchHealth(request);
    test.skip(!h || h.status !== "ok", "FastAPI must be reachable");
    await clearE2ELibrary(request);
  });

  test("empty text file upload shows failure banner", async ({ page }) => {
    await page.goto("/");
    await openSidebarIfMobile(page);
    await page.getByLabel("Choose documents to upload").setInputFiles({
      name: "e2e_empty.txt",
      mimeType: "text/plain",
      buffer: Buffer.alloc(0),
    });
    await expect(
      page.getByText(/empty file|No files could be saved|Upload failed|could not be saved/i).first(),
    ).toBeVisible({ timeout: 30_000 });
  });

  test("API-only upload then UI: needs-sync banner, then Sync indexes", async ({
    page,
    request,
  }) => {
    await request.post("http://127.0.0.1:8000/api/v1/upload", {
      multipart: {
        files: {
          name: CORPUS_NAME,
          mimeType: "text/plain",
          buffer: CORPUS_BUFFER,
        },
      },
    });
    await page.goto("/");
    await openSidebarIfMobile(page);
    await expect(page.getByTitle(CORPUS_NAME)).toBeVisible({ timeout: 30_000 });
    await expect(
      page.getByText(/not indexed for search yet|library is not indexed/i),
    ).toBeVisible({ timeout: 15_000 });
    await clickSyncDocuments(page);
    await waitForSidebarLibraryReady(page, 180_000);
    await expect(
      page.getByText(/not indexed for search yet|library is not indexed/i),
    ).toBeHidden({ timeout: 30_000 });
  });

  test("remove document confirms and clears row", async ({ page, request }) => {
    const dialogDone = page.waitForEvent("dialog").then((d) => d.accept());
    await request.post("http://127.0.0.1:8000/api/v1/upload", {
      multipart: {
        files: {
          name: CORPUS_NAME,
          mimeType: "text/plain",
          buffer: CORPUS_BUFFER,
        },
      },
    });
    await request.post("http://127.0.0.1:8000/api/v1/sync", {
      headers: { "Content-Type": "application/json" },
      data: "{}",
    });
    await page.goto("/");
    await openSidebarIfMobile(page);
    await expect(page.getByTitle(CORPUS_NAME)).toBeVisible({ timeout: 60_000 });
    const removeBtn = page.getByRole("button", { name: new RegExp(`Remove ${CORPUS_NAME}`) });
    await removeBtn.click();
    await dialogDone;
    await expect(page.getByText("No files yet. Upload, then Sync.")).toBeVisible({ timeout: 120_000 });
  });
});

test.describe("C. Grounded chat — requires valid OPENAI_API_KEY", () => {
  test.describe.configure({ mode: "serial" });

  test.beforeEach(async ({ request }) => {
    test.skip(
      !shouldRunLlmSuite(await fetchHealth(request)),
      "Set a non-placeholder OPENAI_API_KEY for grounded chat E2E",
    );
    await clearE2ELibrary(request);
  });

  test("UI upload + sync then JSON API returns corpus token and CEO (positive)", async ({
    page,
    request,
  }) => {
    await page.goto("/");
    await uploadCorpus(page, CORPUS_FILE);
    await waitForLibraryFile(page, CORPUS_NAME, 180_000);
    await waitForSidebarLibraryReady(page, 180_000);
    const ans = await postChatJson(
      request,
      "What is the unique verification token and the CEO codename on record in my synced library? Reply with both exact strings only.",
    );
    expect(ans.text).toContain(TOKEN);
    expect(ans.text).toContain(CEO_ANCHOR);
    expect(ans.mode).toBe("grounded");
    expect(Array.isArray(ans.sources) && ans.sources.length > 0).toBeTruthy();
  });

  test("narrow factual: CEO codename via JSON API (positive)", async ({ page, request }) => {
    await page.goto("/");
    await uploadCorpus(page, CORPUS_FILE);
    await waitForLibraryFile(page, CORPUS_NAME, 180_000);
    await waitForSidebarLibraryReady(page, 180_000);
    const ans = await postChatJson(
      request,
      "What is the CEO codename on record in my library document? One exact token.",
    );
    expect(ans.text).toContain(CEO_ANCHOR);
  });

  test("off-document arithmetic with empty library via JSON API (negative / general)", async ({
    request,
  }) => {
    await clearE2ELibrary(request);
    const ans = await postChatJson(request, "In one short sentence, what is 19 + 23?");
    expect(ans.text).toMatch(/42/);
    expect(ans.mode).not.toBe("grounded");
  });

  test("document-specific question without library must not cite our secret token (negative)", async ({
    request,
  }) => {
    await clearE2ELibrary(request);
    const ans = await postChatJson(
      request,
      `What is the exact string E2E_VERIFY_TOKEN_XY42NZ? If you do not have this in your files, say you cannot find it.`,
    );
    expect(ans.text.toUpperCase().includes(TOKEN)).toBeFalsy();
  });

  test("UI streaming completes for a short reply (sanity)", async ({ page }) => {
    await page.goto("/");
    await openSidebarIfMobile(page);
    await sendChatMessage(page, "Reply with exactly: OK E2E STREAM");
    const text = await waitForAssistantAnyText(page, 90_000);
    expect(text).not.toContain("did not finish");
    expect(text.toUpperCase()).toContain("OK");
  });

  test("task mode Compare via JSON API returns substantive text", async ({ page, request }) => {
    await page.goto("/");
    await uploadCorpus(page, CORPUS_FILE);
    await waitForLibraryFile(page, CORPUS_NAME, 180_000);
    await waitForSidebarLibraryReady(page, 180_000);
    const ans = await postChatJson(
      request,
      "Compare disaster recovery vs revenue mentions in my file in two bullets.",
      "compare",
    );
    expect(ans.text.length).toBeGreaterThan(40);
  });

  test("mobile viewport: open sidebar and see Upload", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await page.getByRole("button", { name: /Open menu/i }).click();
    await expect(page.getByRole("button", { name: /Upload documents/i })).toBeVisible();
  });
});
