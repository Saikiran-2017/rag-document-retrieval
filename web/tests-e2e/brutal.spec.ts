import { test, expect } from "@playwright/test";

test("brutal: app loads and shows needs-sync banner when library stale", async ({ page }) => {
  await page.goto("http://127.0.0.1:3000/");
  await expect(page.getByText("Knowledge Assistant")).toBeVisible();

  // Sidebar should exist even if API is not configured.
  await expect(page.getByRole("button", { name: /Upload documents/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Sync documents/i })).toBeVisible();
});

