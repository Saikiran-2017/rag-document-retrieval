import path from "node:path";
import { test, expect } from "@playwright/test";
import {
  clearE2ELibrary,
  fetchHealth,
  shouldRunLlmSuite,
  waitForAnyDocumentListed,
} from "./helpers/api";
import {
  clickSyncDocuments,
  openSidebarIfMobile,
  sendChatMessage,
  uploadCorpus,
  waitForAssistantNewText,
  waitForLibraryFile,
  waitForSidebarLibraryReady,
  mainTextBubbles,
} from "./helpers/ui";

const CREDILA_FILE = path.join(__dirname, "fixtures", "credila_loan_mock.txt");
const CREDILA_NAME = "credila_loan_mock.txt";

test.describe("Docker web UI smoke", () => {
  test.describe.configure({ mode: "serial" });

  test.beforeEach(async ({ request }) => {
    test.skip(
      !shouldRunLlmSuite(await fetchHealth(request)),
      "Requires API + non-placeholder OPENAI_API_KEY for grounded UI smoke",
    );
    await clearE2ELibrary(request);
  });

  test("upload → sync → broad → field/value → metadata → negative; citations render", async ({
    page,
    request,
  }) => {
    test.setTimeout(240_000);

    await page.goto("/", { waitUntil: "domcontentloaded" });
    await openSidebarIfMobile(page);

    // Upload via the same UI path as users. (Helper uses the file picker input.)
    await uploadCorpus(page, CREDILA_FILE);
    const storedName = await waitForAnyDocumentListed(request, 60_000);
    // Best-effort UI assertion (backend may sanitize stored filename).
    await waitForLibraryFile(page, storedName || CREDILA_NAME, 180_000);

    // Sync (indexes) via UI.
    await clickSyncDocuments(page);
    await waitForSidebarLibraryReady(page, 180_000);

    const ask = async (q: string) => {
      const prev = await mainTextBubbles(page).count();
      await sendChatMessage(page, q);
      return await waitForAssistantNewText(page, prev, 120_000);
    };

    // Broad document summary (grounded + citations)
    const broad = await ask("what is this document about?");
    expect(broad.toLowerCase()).toContain("document");
    expect(broad).toMatch(/\[SOURCE\s+\d+\]/i);

    // Field/value
    const amt = await ask("how much is my loan?");
    expect(amt).toMatch(/1,026,904\.00/);
    expect(amt).toMatch(/\[SOURCE\s+\d+\]/i);

    // Metadata question
    const meta = await ask("what is the file name?");
    expect(meta.toLowerCase()).toContain(".txt");
    expect(meta).toMatch(/\[SOURCE\s+\d+\]/i);

    // Negative question (must not hallucinate, should refuse based on docs)
    const neg = await ask("what is the recipe for chocolate cake in my uploaded files?");
    expect(neg).toMatch(/I don't know based on the provided documents/i);
  });
});

