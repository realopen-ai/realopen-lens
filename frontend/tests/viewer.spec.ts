import { test, expect, type Page } from "@playwright/test";

async function setupViewer(
  page: Page,
  options: {
    failOriginal?: boolean;
    failDetails?: boolean;
    librarySource?: boolean;
  } = {},
) {
  const a = "a".repeat(64),
    b = "b".repeat(64);
  const group = {
    id: "d".repeat(32),
    name: "Saved images",
    collection_id: "library-v1",
    revision: 1,
    member_image_ids: [] as string[],
  };
  const requests: { path: string; body: any }[] = [];
  const records = [a, b].map((id, index) => ({
    rank: index + 1,
    image_id: id,
    paths: [`/photos/${index ? "green" : "red"}.png`],
    width: 640,
    height: 480,
    text: "Short preview",
    status: "ready",
    ...(options.librarySource
      ? {
          sources: [
            {
              collection: "other folder",
              collection_id: "f".repeat(32),
              generation: "1".repeat(32),
              image_id: id,
              paths: [`/photos/${index ? "green" : "red"}.png`],
              siglip_precision: "fp32",
            },
          ],
        }
      : {}),
    issues: [],
    visual_similarity: null,
  }));
  await page.route("**/api/**", async (route) => {
    const request = route.request(),
      url = new URL(request.url()),
      path = url.pathname;
    const json = (data: unknown, status = 200) =>
      route.fulfill({ status, json: data });
    if (request.method() !== "GET")
      requests.push({ path, body: request.postDataJSON() });
    if (path === "/api/session") return json({ allowed_roots: ["/photos"] });
    if (path === "/api/collections")
      return json([
        {
          id: "c".repeat(32),
          name: "personal",
          root: "/photos",
          generation: "e".repeat(32),
          siglip_precision: "int8",
        },
      ]);
    if (path === "/api/tasks") return json([]);
    if (path === "/api/virtual-collections") return json([]);
    if (path.endsWith("/images"))
      return json({ results: records, total: 2, generation: "e".repeat(32) });
    if (path === "/api/image-details") {
      if (options.failDetails)
        return json({ detail: "Source folder is unavailable" }, 400);
      const id = url.searchParams.get("image_id")!;
      return json({
        image_id: id,
        collection: "personal",
        collection_id: "c".repeat(32),
        generation: "e".repeat(32),
        paths: records.find((row) => row.image_id === id)!.paths,
        available_paths: ["/photos/red.png"],
        metadata: { format: "PNG" },
        text:
          `Full OCR for ${id === a ? "red" : "green"}. ` +
          "Texte français مرحبا ".repeat(40),
        groups: [group],
      });
    }
    if (path === "/api/image-actions") return json({ opened: true });
    if (path.endsWith("/members")) {
      group.member_image_ids =
        request.method() === "DELETE" ? [] : request.postDataJSON().image_ids;
      group.revision++;
      return json(group);
    }
    if (path.endsWith("/original") && options.failOriginal)
      return json({ detail: "Original unavailable" }, 404);
    if (path.endsWith("/original") || path.endsWith("/thumbnail"))
      return route.fulfill({
        contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#dbe5cc"/><rect x="100" y="60" width="440" height="360" rx="12" fill="#fff"/><text x="140" y="130" fill="#23332c" font-size="32">Invoice 004231</text><path d="M140 180h340M140 230h250M140 280h340M140 330h180" stroke="#a8bf9a" stroke-width="12"/></svg>',
      });
    return json({ detail: "Unexpected endpoint" }, 404);
  });
  await page.goto("/");
  const trigger = page.getByRole("button", {
    name: "View red.png",
    exact: true,
  });
  await trigger.click();
  return { requests, trigger, dialog: page.getByRole("dialog") };
}

test("viewer navigates, zooms, traps focus and restores focus in both themes", async ({
  page,
}, info) => {
  const { dialog, trigger } = await setupViewer(page);
  await expect(dialog.getByRole("heading", { name: "red.png" })).toBeVisible();
  await expect(dialog.locator("pre")).toContainText("Texte français مرحبا");
  await expect(
    dialog.getByRole("button", { name: "Previous image" }),
  ).toBeDisabled();
  const fitWidth = (await dialog.locator(".viewer-picture").boundingBox())!
    .width;
  await dialog.getByRole("button", { name: "Zoom in" }).click();
  await expect(dialog.locator("output")).toHaveText("100%");
  expect(
    (await dialog.locator(".viewer-picture").boundingBox())!.width,
  ).toBeGreaterThanOrEqual(fitWidth);
  await dialog.getByRole("button", { name: "Zoom in" }).click();
  await expect(dialog.locator(".viewer-canvas")).toHaveClass(/zoom-2/);
  await dialog.getByRole("button", { name: "Next image" }).click();
  await expect(
    dialog.getByRole("heading", { name: "green.png" }),
  ).toBeVisible();
  await expect(dialog.locator("output")).toHaveText("Fit");
  await expect(
    dialog.getByRole("button", { name: "Next image" }),
  ).toBeDisabled();
  await page.keyboard.press("ArrowLeft");
  await expect(dialog.getByRole("heading", { name: "red.png" })).toBeVisible();
  await dialog.getByRole("button", { name: "Close details" }).focus();
  await page.keyboard.press("Shift+Tab");
  await expect(dialog.locator("pre")).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(
    dialog.getByRole("button", { name: "Close details" }),
  ).toBeFocused();
  await dialog.locator(".viewer-info").evaluate((node) => (node.scrollTop = 0));
  await dialog.locator(".viewer-body").evaluate((node) => (node.scrollTop = 0));
  for (const theme of ["light", "dark"]) {
    await page.evaluate(
      (value) => (document.documentElement.dataset.theme = value),
      theme,
    );
    await page.screenshot({
      path: `../runs/image-viewer-v1/${info.project.name}-${theme}.png`,
      animations: "disabled",
    });
    expect(
      await dialog.evaluate((node) => node.scrollWidth <= node.clientWidth),
    ).toBeTruthy();
  }
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
});

test("viewer opens verified originals, reveals sources and updates memberships", async ({
  page,
}) => {
  const { dialog, requests } = await setupViewer(page);
  await dialog
    .getByRole("button", { name: "Open original", exact: true })
    .click();
  await expect(dialog.locator('p[role="status"]')).toContainText(
    "Opened in your default image app",
  );
  await dialog.getByRole("button", { name: "Reveal in folder" }).click();
  await expect
    .poll(() =>
      requests
        .filter((r) => r.path === "/api/image-actions")
        .map((r) => r.body.action),
    )
    .toEqual(["open", "reveal"]);
  await dialog.getByRole("checkbox", { name: "Saved images" }).click();
  await expect(
    dialog.getByRole("checkbox", { name: "Saved images" }),
  ).toBeChecked();
  await dialog.getByRole("checkbox", { name: "Saved images" }).click();
  await expect(
    dialog.getByRole("checkbox", { name: "Saved images" }),
  ).not.toBeChecked();
  await expect(
    dialog.getByRole("button", { name: "Relevant", exact: true }),
  ).toBeDisabled();
  expect(requests.find((r) => r.path.endsWith("/members"))!.body).toEqual({
    collection: "personal",
    all_folders: true,
    revision: 1,
    image_ids: ["a".repeat(64)],
  });
});

test("viewer keeps a usable thumbnail and visible errors when sources are unavailable", async ({
  page,
}) => {
  const { dialog } = await setupViewer(page, {
    failOriginal: true,
    failDetails: true,
  });
  await expect(dialog.getByRole("alert")).toContainText(
    "Source folder is unavailable",
  );
  await expect(dialog.locator('p[role="status"]')).toContainText(
    "Showing the indexed thumbnail",
  );
  await expect(
    dialog.getByRole("button", { name: "Open original", exact: true }),
  ).toBeDisabled();
  await expect(dialog.getByAltText("red.png")).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await expect(
    dialog.getByRole("heading", { name: "green.png" }),
  ).toBeVisible();
});

test("viewer preserves stable source identity and generation in library results", async ({
  page,
}) => {
  const { dialog, requests } = await setupViewer(page, { librarySource: true });
  await dialog
    .getByRole("button", { name: "Open original", exact: true })
    .click();
  await expect
    .poll(() => requests.find((r) => r.path === "/api/image-actions")?.body)
    .toEqual({
      collection: "other folder",
      collection_id: "f".repeat(32),
      generation: "1".repeat(32),
      image_id: "a".repeat(64),
      action: "open",
    });
});

test("stale membership requires explicit retry with the refreshed revision", async ({
  page,
}) => {
  const { dialog } = await setupViewer(page);
  let attempts = 0;
  await page.route("**/api/virtual-collections/*/members", async (route) => {
    attempts++;
    await route.fulfill({
      status: 409,
      json: { detail: "Virtual collection changed; refresh and retry" },
    });
  });
  await dialog.getByRole("checkbox", { name: "Saved images" }).click();
  await expect(dialog.getByRole("alert")).toContainText("refresh and retry");
  await expect(
    dialog.getByRole("checkbox", { name: "Saved images" }),
  ).not.toBeChecked();
  expect(attempts).toBe(1);
});
