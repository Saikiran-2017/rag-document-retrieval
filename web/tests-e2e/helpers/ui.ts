import type { Page } from "@playwright/test";
import { expect } from "@playwright/test";

export async function withStage<T>(stage: string, fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`E2E stage failed: ${stage}\n${msg}`);
  }
}

export async function openSidebarIfMobile(page: Page): Promise<void> {
  const menu = page.getByRole("button", { name: /Open menu/i });
  if (await menu.isVisible().catch(() => false)) {
    await menu.click();
  }
}

export async function uploadCorpus(page: Page, absolutePath: string): Promise<void> {
  await openSidebarIfMobile(page);
  await page.getByLabel("Choose documents to upload").setInputFiles(absolutePath);
}

/** Wait until a filename appears in the library list (after upload + auto-sync in UI). */
export async function waitForLibraryFile(
  page: Page,
  filename: string,
  timeout = 180_000,
): Promise<void> {
  await openSidebarIfMobile(page);
  // Avoid strict-mode collisions when the filename appears both as a row title and in a path hint.
  const sidebar = page.locator("#app-sidebar");
  const byRowTitle = sidebar.locator(`[title="${filename}"]`).first();
  const byExactText = sidebar.getByText(filename, { exact: true }).first();
  await expect(byRowTitle.or(byExactText)).toBeVisible({ timeout });
}

export async function clickSyncDocuments(page: Page): Promise<void> {
  await openSidebarIfMobile(page);
  await page.getByRole("button", { name: /Sync documents/i }).click();
}

/** Sidebar health pill: exact "Ready" or "Ready · limited" (not substring-only matches). */
export async function waitForSidebarLibraryReady(page: Page, timeout = 180_000): Promise<void> {
  await openSidebarIfMobile(page);
  await expect(page.locator("#app-sidebar").getByText(/^Ready/)).toBeVisible({ timeout });
}

export async function sendChatMessage(page: Page, text: string): Promise<void> {
  await page.getByPlaceholder("Message…").fill(text);
  await page.getByRole("button", { name: /^Send$/ }).click();
}

export function mainTextBubbles(page: Page) {
  return page.locator("main .whitespace-pre-wrap");
}

export function assistantBubbles(page: Page) {
  // Assistant bubbles are in the left-aligned message cards and contain the body wrapper.
  // This is intentionally looser than role-based selectors to keep tests resilient to markup changes.
  return page.locator("main .whitespace-pre-wrap");
}

/**
 * Wait for the latest assistant bubble to contain text (non-streaming) and optionally a substring.
 */
export async function waitForAssistantGrounded(
  page: Page,
  mustContain: string,
  timeout = 120_000,
): Promise<void> {
  const bubbles = mainTextBubbles(page);
  await expect(bubbles.last()).not.toHaveText(/^\s*$/, { timeout });
  await expect(bubbles.last()).toContainText(mustContain, { timeout });
}

export async function waitForAssistantAnyText(page: Page, timeout = 120_000): Promise<string> {
  const bubbles = mainTextBubbles(page);
  await expect(bubbles.last()).not.toHaveText(/^\s*$/, { timeout });
  return (await bubbles.last().innerText()).trim();
}

export async function waitForAssistantNewText(
  page: Page,
  prevBubbleCount: number,
  timeout = 120_000,
): Promise<string> {
  const bubbles = mainTextBubbles(page);
  await expect(bubbles).toHaveCount(prevBubbleCount + 1, { timeout });
  await expect(bubbles.last()).not.toHaveText(/^\s*$/, { timeout });
  return (await bubbles.last().innerText()).trim();
}

/** Documents mode chip (latest occurrence in main chat). */
export async function latestAssistantHasDocumentsBadge(page: Page): Promise<boolean> {
  const badge = page.locator("main").getByText("Documents", { exact: true }).last();
  return badge.isVisible().catch(() => false);
}
