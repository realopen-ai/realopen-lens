import { test, expect, type Page } from "@playwright/test";

import type { Collection } from "../src/api";

async function setup(page: Page, automaticTask: boolean | "completed" = false) {
  let folders: Collection[] = [
    {
      id: "c".repeat(32),
      name: "personal",
      root: "/photos",
      generation: "g",
      siglip_precision: "fp32",
    },
  ];
  const requests: { path: string; body: Record<string, string> }[] = [];
  const image = {
    rank: 1,
    image_id: "a".repeat(64),
    paths: ["/photos/red.png"],
    thumbnail: null,
    width: 64,
    height: 48,
    text: "",
    status: "ready",
    issues: [],
  };
  const task = {
    id: "refresh",
    origin: automaticTask ? "automatic" : "manual",
    kind: "index",
    collection: "personal",
    status: automaticTask === "completed" ? "completed" : "queued",
    phase: "queued",
    progress: {},
    revision: 0,
    elapsed_seconds: 0,
    error: null,
    result: null,
  };
  let taskPolls = 0;
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/session")
      return route.fulfill({ json: { allowed_roots: ["/photos"] } });
    if (path === "/api/collections") return route.fulfill({ json: folders });
    if (path === "/api/virtual-collections") return route.fulfill({ json: [] });
    if (path === "/api/tasks")
      return route.fulfill({
        json: automaticTask && taskPolls++ > 0 ? [task] : [],
      });
    if (path.endsWith("/images"))
      return route.fulfill({
        json: {
          results: [image],
          total: automaticTask === "completed" && taskPolls > 1 ? 2 : 1,
          generation: "g",
        },
      });
    if (
      path.startsWith("/api/collections/") &&
      route.request().method() === "POST"
    ) {
      const body = route.request().postDataJSON();
      requests.push({ path, body });
      if (path.endsWith("/auto-refresh")) {
        const settings = {
          enabled: body.enabled,
          interval_seconds: body.interval_seconds,
          state: body.enabled ? "waiting" : "off",
          error: null,
          last_checked_at: null,
          task_id: null,
        };
        folders = [{ ...folders[0], auto_refresh: settings }];
        return route.fulfill({ json: settings });
      }
      if (path.endsWith("/rename")) {
        if (body.name === "taken")
          return route.fulfill({
            status: 409,
            json: { detail: "That folder name is already in use" },
          });
        folders = [{ ...folders[0], name: body.name }];
        return route.fulfill({ json: { name: body.name } });
      }
      if (path.endsWith("/remove")) {
        folders = [];
        return route.fulfill({ json: { name: null } });
      }
      return route.fulfill({ status: 202, json: task });
    }
    if (path === "/api/tasks/refresh") return route.fulfill({ json: task });
    if (path.endsWith("/events"))
      return route.fulfill({
        contentType: "text/event-stream",
        body: `event: progress\ndata: ${JSON.stringify({ ...task, status: "running", phase: "vision", elapsed_seconds: 2, revision: 1, progress: { unique_images: 2, phase_completed_images: 1 } })}\n\n`,
      });
    return route.fulfill({ json: {} });
  });
  await page.goto("/");
  return requests;
}

test("rename keeps the dialog on conflict and switches to the renamed folder", async ({
  page,
}) => {
  const requests = await setup(page);
  await page.getByRole("button", { name: "Manage folder personal" }).click();
  await page.getByRole("button", { name: "Rename", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Rename personal" });
  await dialog.getByLabel("Folder name").fill("taken");
  await dialog.getByRole("button", { name: "Save name" }).click();
  await expect(dialog.getByRole("alert")).toContainText("already in use");
  await dialog.getByLabel("Folder name").fill("Mes photos صور");
  await dialog.getByRole("button", { name: "Save name" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Manage folder Mes photos صور" }),
  ).toBeVisible();
  expect(requests[1].body.collection_id).toBe("c".repeat(32));
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Manage folder Mes photos صور" }),
  ).toBeVisible();
});

test("remove requires confirmation, supports cancellation, and clears the last folder", async ({
  page,
}, info) => {
  const requests = await setup(page);
  const open = async () => {
    await page.getByRole("button", { name: "Manage folder personal" }).click();
    await page
      .getByRole("button", { name: "Remove from Lens", exact: true })
      .click();
  };
  await open();
  let dialog = page.getByRole("dialog", { name: "Remove personal from Lens" });
  await expect(dialog).toContainText("Your original files stay untouched");
  await page.screenshot({
    path: `../runs/folder-removal-${info.project.name}.png`,
    fullPage: true,
  });
  await dialog.getByRole("button", { name: "Cancel" }).click();
  expect(requests).toHaveLength(0);
  await open();
  dialog = page.getByRole("dialog", { name: "Remove personal from Lens" });
  await dialog
    .getByRole("button", { name: "Remove from Lens", exact: true })
    .click();
  await expect(dialog).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Manage folder personal" }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Your collections will appear here."),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Make room for a little discovery." }),
  ).toBeVisible();
  expect(requests).toHaveLength(1);
});

test("refresh displays live indexing progress and disables conflicting actions", async ({
  page,
}, info) => {
  const requests = await setup(page);
  await page.getByRole("button", { name: "Manage folder personal" }).click();
  await page.getByRole("button", { name: "Refresh index" }).click();
  const progress = page.getByRole("region", { name: "Task progress" });
  await expect(progress).toContainText("Indexing");
  await expect(progress).toContainText("1/2 images");
  await page.getByRole("button", { name: "Manage folder personal" }).click();
  await expect(
    page.getByRole("button", { name: "Rename", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Refresh index" }),
  ).toBeDisabled();
  expect(requests[0]).toEqual({
    path: "/api/collections/refresh",
    body: { collection: "personal", collection_id: "c".repeat(32) },
  });
  await page.screenshot({
    path: `../runs/folder-management-${info.project.name}.png`,
    fullPage: true,
  });
});

test("automatic refresh is opt-in, styled, configurable and retained after reload", async ({
  page,
}, info) => {
  const requests = await setup(page);
  await page.getByRole("button", { name: "Manage folder personal" }).click();
  const checkbox = page.getByRole("checkbox", { name: "Automatic refresh" });
  await expect(checkbox).not.toBeChecked();
  await checkbox.check();
  await expect(page.getByText("Watching for changes")).toBeVisible();
  await page.getByLabel("Refresh interval for personal").selectOption("300");
  await expect.poll(() => requests.at(-1)?.body.interval_seconds).toBe(300);
  expect(requests.at(-1)?.body).toEqual({
    collection: "personal",
    collection_id: "c".repeat(32),
    enabled: true,
    interval_seconds: 300,
  });
  await page.screenshot({
    path: `../runs/automatic-refresh-${info.project.name}.png`,
    fullPage: true,
  });
  await page.reload();
  await page.getByRole("button", { name: "Manage folder personal" }).click();
  await expect(checkbox).toBeChecked();
  await expect(page.getByLabel("Refresh interval for personal")).toHaveValue(
    "300",
  );
  await checkbox.uncheck();
  await expect(page.getByLabel("Refresh interval for personal")).toHaveCount(0);
});

test("new automatic jobs are discovered without reloading and stream progress", async ({
  page,
}) => {
  await setup(page, true);
  await expect(
    page.getByRole("region", { name: "Task progress" }),
  ).toContainText("Refreshing automatically", { timeout: 10000 });
  await expect(
    page.getByRole("region", { name: "Task progress" }),
  ).toContainText("1/2 images");
  await page.getByRole("button", { name: "Manage folder personal" }).click();
  // Future automatic work can be disabled while the existing task is still running.
  const checkbox = page.getByRole("checkbox", { name: "Automatic refresh" });
  await expect(checkbox).toBeEnabled();
  await checkbox.check();
  await expect(
    page.getByRole("button", { name: "Refresh index" }),
  ).toBeDisabled();
});

test("an automatic refresh finishing between discovery polls updates the gallery", async ({
  page,
}) => {
  await setup(page, "completed");
  await expect(
    page.getByText("1 images in this collection", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("2 images in this collection", { exact: true }),
  ).toBeVisible({ timeout: 10000 });
  await page.getByRole("button", { name: "Activity", exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Task history" }),
  ).toContainText("automatic refresh");
});

test("automatic completion keeps the current tool open", async ({ page }) => {
  await setup(page, "completed");
  await page.getByRole("button", { name: "Activity", exact: true }).click();
  const history = page.getByRole("region", { name: "Task history" });
  await expect(history).toContainText("automatic refresh", { timeout: 10000 });
  await expect(
    page.getByRole("heading", { name: "Activity", exact: true }),
  ).toBeVisible();
});

for (const theme of ["light", "dark"] as const) {
  test(`folder popover stays visible outside a short folder list in ${theme} mode`, async ({
    page,
  }, info) => {
    await page.emulateMedia({ colorScheme: theme });
    // Match the packaged app's style policy: positioning must work without unsafe-inline.
    await page.route("http://127.0.0.1:4173/", async (route) => {
      const response = await route.fetch();
      await route.fulfill({
        response,
        headers: {
          ...response.headers(),
          "Content-Security-Policy":
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' blob:",
        },
      });
    });
    await setup(page);
    await page.locator(".collection-list").evaluate((el) => {
      el.style.maxHeight = "50px";
    });
    const trigger = page.getByRole("button", {
      name: "Manage folder personal",
      exact: true,
    });
    await trigger.click();
    const menu = page.getByRole("dialog", {
      name: "Actions for personal",
      exact: true,
    });
    await expect(
      menu.getByRole("button", { name: "Remove from Lens", exact: true }),
    ).toBeVisible();
    await menu
      .getByRole("checkbox", { name: "Automatic refresh", exact: true })
      .check();
    await expect(
      menu.getByLabel("Refresh interval for personal"),
    ).toBeVisible();
    const bounds = await menu.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.y).toBeGreaterThanOrEqual(0);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(
      page.viewportSize()!.height,
    );
    await expect(
      menu.getByRole("button", { name: "Refresh index", exact: true }),
    ).toHaveCSS("font-size", "12px");
    await page.screenshot({
      path: `../runs/folder-popover-${theme}-${info.project.name}.png`,
      fullPage: true,
    });
    await page.keyboard.press("Escape");
    await expect(menu).toHaveCount(0);
    await expect(trigger).toBeFocused();
    await trigger.click();
    await page.mouse.click(
      page.viewportSize()!.width - 8,
      page.viewportSize()!.height - 8,
    );
    await expect(menu).toHaveCount(0);
  });
}

test("indexing progress distinguishes cached images from newly processed work", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/tasks/refresh/events", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body: `event: progress\ndata: ${JSON.stringify({ id: "refresh", kind: "index", collection: "personal", status: "running", phase: "ocr", elapsed_seconds: 1, revision: 1, error: null, result: null, progress: { unique_images: 105, phase_completed_images: 104, phase_cached_images: 104, phase_processed_images: 0 } })}\n\n`,
    }),
  );
  await page
    .getByRole("button", { name: "Manage folder personal", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Refresh index", exact: true })
    .click();
  const progress = page.getByRole("region", { name: "Task progress" });
  await expect(progress).toContainText("0 processed · 104 cached · 105 total");
  await expect(
    page.getByRole("progressbar", { name: "Stage progress" }),
  ).toHaveAttribute("value", "104");
});
