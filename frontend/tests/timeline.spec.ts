import { test, expect, type Page } from "@playwright/test";

async function setup(page: Page) {
  const requests: URL[] = [];
  let conflict = false;
  const images = Array.from({ length: 14 }, (_, i) => ({
    rank: i + 1,
    image_id: (i + 1).toString(16).padStart(64, "0"),
    paths: [`/photos/image-${i + 1}.png`],
    thumbnail: "thumb",
    width: 640,
    height: 480,
    text: "",
    status: "ready",
    issues: [],
    visual_similarity: null,
    timeline: {
      date: i === 13 ? null : "2024-06-15T10:00:00+02:00",
      day: i === 13 ? null : "2024-06-15",
      month: i === 13 ? null : "2024-06",
      source: i === 13 ? "unknown" : i % 2 ? "modified" : "capture",
    },
  }));
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url()),
      path = url.pathname;
    if (path === "/api/session")
      return route.fulfill({ json: { allowed_roots: ["/photos"] } });
    if (path === "/api/collections")
      return route.fulfill({
        json: [
          {
            id: "c".repeat(32),
            name: "personal",
            root: "/photos",
            generation: "g",
            siglip_precision: "int8",
          },
        ],
      });
    if (path === "/api/virtual-collections")
      return route.fulfill({
        json: [
          {
            id: "d".repeat(32),
            name: "Favorites",
            revision: 1,
            collection_id: "c".repeat(32),
            member_count: 1,
          },
        ],
      });
    if (path === "/api/tasks") return route.fulfill({ json: [] });
    if (path.endsWith("/images")) {
      requests.push(url);
      const offset = Number(url.searchParams.get("offset") || "0");
      if (offset && conflict)
        return route.fulfill({
          status: 409,
          json: { detail: "Image timeline changed; reload images to continue" },
        });
      const rows = path.includes("virtual-collections")
        ? [images[0]]
        : url.searchParams.get("order") === "oldest"
          ? [...images.slice(0, 13)].reverse().concat(images[13])
          : images;
      return route.fulfill({
        json: {
          results: rows.slice(offset, offset + 12),
          total: rows.length,
          next_offset: offset + 12 < rows.length ? offset + 12 : null,
          generation: "g",
          snapshot: `${path}:${url.searchParams.get("order")}`,
        },
      });
    }
    if (path === "/api/image-details") {
      const row = images.find(
        (image) => image.image_id === url.searchParams.get("image_id"),
      )!;
      return route.fulfill({
        json: {
          image_id: row.image_id,
          collection: "personal",
          collection_id: "c".repeat(32),
          generation: "g",
          paths: row.paths,
          available_paths: row.paths,
          metadata: {},
          text: "",
          groups: [],
        },
      });
    }
    if (path.endsWith("/thumbnail") || path.endsWith("/original"))
      return route.fulfill({
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
          "base64",
        ),
      });
    return route.fulfill({ json: {} });
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "View image-1.png", exact: true }),
  ).toBeVisible();
  return {
    requests,
    conflict: () => {
      conflict = true;
    },
  };
}

test("scroll paging preserves selection and continuous date groups", async ({
  page,
}, info) => {
  const { requests } = await setup(page);
  await page
    .getByRole("checkbox", { name: "Select image-1.png", exact: true })
    .check();
  await page.locator(".load-more").scrollIntoViewIfNeeded();
  await expect(page.locator(".image-card")).toHaveCount(14);
  await expect(
    page.getByRole("checkbox", { name: "Select image-1.png", exact: true }),
  ).toBeChecked();
  await expect(
    page.getByRole("heading", { name: "June 2024", exact: true }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("heading", { name: "Saturday, Jun 15", exact: true }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("heading", { name: "Date unknown", exact: true }),
  ).toHaveCount(1);
  await expect(
    page.locator(".timeline-date").filter({ hasText: "File modified" }).first(),
  ).toHaveCount(1);
  const next = requests.find((url) => url.searchParams.get("offset") === "12")!;
  expect(next.searchParams.get("snapshot")).toBe(
    "/api/collections/personal/images:newest",
  );
  expect(next.searchParams.get("generation")).toBe("g");
  await expect(
    page.getByRole("button", { name: "Load more images" }),
  ).toHaveCount(0);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: `../runs/timeline-${info.project.name}.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("sorting restarts paging and library and virtual browsing preserve scope", async ({
  page,
}) => {
  const { requests } = await setup(page);
  await page.getByLabel("Sort images", { exact: true }).selectOption("oldest");
  await expect(
    page
      .locator(".image-card")
      .first()
      .getByRole("button", { name: "View image-13.png", exact: true }),
  ).toBeVisible();
  expect(requests.at(-1)!.searchParams.has("snapshot")).toBe(false);
  await page.getByLabel("Search in", { exact: true }).selectOption("");
  await expect
    .poll(() => requests.at(-1)!.pathname)
    .toBe("/api/library/images");
  expect(requests.at(-1)!.searchParams.get("order")).toBe("oldest");
  await page.getByLabel("Search in", { exact: true }).selectOption("personal");
  await page
    .getByRole("button", {
      name: "Open virtual collection Favorites",
      exact: true,
    })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(1);
  expect(requests.at(-1)!.pathname).toContain("/virtual-collections/");
  expect(requests.at(-1)!.searchParams.get("all_folders")).toBe("false");
  expect(requests.at(-1)!.searchParams.has("snapshot")).toBe(false);
});

test("stale pages stop automatic retries and offer an explicit reload", async ({
  page,
}) => {
  const control = await setup(page);
  control.conflict();
  await page.locator(".load-more").scrollIntoViewIfNeeded();
  await expect(
    page.getByRole("button", { name: "Reload images", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".image-card")).toHaveCount(12);
  const count = control.requests.length;
  await page.getByLabel("Sort images", { exact: true }).focus();
  expect(control.requests.length).toBe(count);
  await page
    .getByRole("button", { name: "Reload images", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Reload images", exact: true }),
  ).toHaveCount(0);
  expect(control.requests.at(-1)!.searchParams.get("offset")).toBe("0");
});

test("viewer navigation follows the appended timeline order", async ({
  page,
}) => {
  await setup(page);
  await page.locator(".load-more").scrollIntoViewIfNeeded();
  await expect(page.locator(".image-card")).toHaveCount(14);
  await page
    .getByRole("button", { name: "View image-12.png", exact: true })
    .click();
  const viewer = page.getByRole("dialog");
  await expect(
    viewer.getByRole("heading", { name: "image-12.png", exact: true }),
  ).toBeVisible();
  await viewer.getByRole("button", { name: "Next image", exact: true }).click();
  await expect(
    viewer.getByRole("heading", { name: "image-13.png", exact: true }),
  ).toBeVisible();
  await viewer
    .getByRole("button", { name: "Close details", exact: true })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(14);
});

test("background refresh retains library scope and the selected date order", async ({
  page,
}) => {
  const { requests } = await setup(page);
  await page.getByLabel("Sort images", { exact: true }).selectOption("oldest");
  await page.getByLabel("Search in", { exact: true }).selectOption("");
  await expect
    .poll(() => requests.at(-1)!.pathname)
    .toBe("/api/library/images");
  const task = {
    id: "auto",
    kind: "index",
    collection: "another-folder",
    origin: "automatic",
    revision: 1,
    status: "completed",
    phase: "finished",
    elapsed_seconds: 1,
    progress: {},
    error: null,
    result: {},
  };
  await page.route("**/api/tasks?summary=true", (route) =>
    route.fulfill({ json: [task] }),
  );
  await page.route("**/api/tasks/auto", (route) =>
    route.fulfill({ json: task }),
  );
  const count = requests.length;
  await expect
    .poll(() => requests.length, { timeout: 10000 })
    .toBeGreaterThan(count);
  expect(requests.at(-1)!.pathname).toBe("/api/library/images");
  expect(requests.at(-1)!.searchParams.get("order")).toBe("oldest");
  expect(requests.at(-1)!.searchParams.get("offset")).toBe("0");
});
