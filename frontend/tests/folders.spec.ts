import { test, expect, type Page } from "@playwright/test";

async function setup(page: Page) {
  let folders = [
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
    kind: "index",
    collection: "personal",
    status: "queued",
    phase: "queued",
    progress: {},
    revision: 0,
    elapsed_seconds: 0,
    error: null,
    result: null,
  };
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/session")
      return route.fulfill({ json: { allowed_roots: ["/photos"] } });
    if (path === "/api/collections") return route.fulfill({ json: folders });
    if (path === "/api/virtual-collections" || path === "/api/tasks")
      return route.fulfill({ json: [] });
    if (path.endsWith("/images"))
      return route.fulfill({
        json: { results: [image], total: 1, generation: "g" },
      });
    if (
      path.startsWith("/api/collections/") &&
      route.request().method() === "POST"
    ) {
      const body = route.request().postDataJSON();
      requests.push({ path, body });
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
