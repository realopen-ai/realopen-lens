import { test, expect, type Page } from "@playwright/test";

async function setup(page: Page, expired = false) {
  const rows = Array.from({ length: 4 }, (_, i) => ({
    rank: i + 1,
    image_id: String(i + 1).repeat(64),
    paths: [`/photos/image-${i + 1}.png`],
    width: 640,
    height: 480,
    thumbnail: null,
    text: "",
    visual_similarity: 0.8,
    status: "ready",
    issues: [],
  }));
  const searches: unknown[] = [],
    labels: string[] = [];
  await page.route("**/api/**", (route) => {
    const request = route.request(),
      path = new URL(request.url()).pathname;
    if (path.endsWith("/original") || path.endsWith("/thumbnail"))
      return route.fulfill({
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
          "base64",
        ),
      });
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
    if (path.endsWith("/images"))
      return route.fulfill({
        json: { results: [rows[0]], total: 1, generation: "g" },
      });
    if (path === "/api/tasks/search" || path === "/api/tasks/similar") {
      searches.push(request.postDataJSON());
      return route.fulfill({
        json: {
          id: "search",
          kind: path.endsWith("similar") ? "similar" : "search",
          collection: "personal",
          status: "completed",
          revision: 1,
          phase: "finished",
          elapsed_seconds: 1,
          progress: {},
          result: {
            results: rows.slice(0, 2),
            total: 4,
            next_cursor: "cursor",
            run_id: "saved",
            ranking: { kind: "hybrid", sha256: null },
            feedback: { enabled: true, applied: false },
          },
        },
      });
    }
    if (path === "/api/search-pages")
      return route.fulfill(
        expired
          ? {
              status: 409,
              json: { detail: "Search scope changed; run the search again" },
            }
          : {
              json: {
                results: rows.slice(2),
                total: 4,
                next_cursor: null,
                run_id: "saved",
              },
            },
      );
    if (path.startsWith("/api/feedback/runs/")) {
      if (request.method() === "PUT") labels.push(path);
      return route.fulfill({
        json: {
          judgments: [],
          revision: 1,
          memory:
            request.method() === "PUT"
              ? {
                  image_id: path.split("/").at(-1),
                  label: request.postDataJSON().label,
                  event_id: 1,
                  run_id: "saved",
                }
              : null,
        },
      });
    }
    return route.fulfill({ json: [] });
  });
  await page.goto("/");
  await expect(page.locator(".image-card")).toHaveCount(1);
  return { searches, labels };
}

test("search pages append once and later images retain feedback controls", async ({
  page,
}, info) => {
  const state = await setup(page);
  await page.getByLabel("Search images", { exact: true }).fill("sunset");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(page.locator(".image-card")).toHaveCount(2);
  expect(state.searches[0]).toMatchObject({ paginate: true, limit: 50 });
  await page
    .getByRole("button", { name: "Load more results", exact: true })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(4);
  await expect(
    page.getByRole("button", { name: "Load more results", exact: true }),
  ).toHaveCount(0);
  await page
    .locator(".image-card")
    .nth(3)
    .getByRole("button", { name: "Mark image-4.png relevant", exact: true })
    .click();
  expect(state.labels[0]).toContain(`/saved/${"4".repeat(64)}`);
  expect(state.searches).toHaveLength(1);
  await page.screenshot({
    path: `../runs/search-pagination-${info.project.name}.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("stale search page reports retry without mixing or clearing existing images", async ({
  page,
}) => {
  await setup(page, true);
  await page.getByLabel("Search images", { exact: true }).fill("sunset");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page
    .getByRole("button", { name: "Load more results", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText("Search scope changed");
  await expect(page.locator(".image-card")).toHaveCount(2);
});

test("image similarity uses the same search paging workflow", async ({
  page,
}) => {
  const state = await setup(page);
  await page.getByRole("button", { name: "Similar", exact: true }).click();
  await expect(page.locator(".image-card")).toHaveCount(2);
  expect(state.searches[0]).toMatchObject({
    paginate: true,
    limit: 50,
    image_id: "1".repeat(64),
  });
  await page
    .getByRole("button", { name: "Load more results", exact: true })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(4);
});

test("hiding the complete visible page still allows loading remaining results", async ({
  page,
}) => {
  await setup(page);
  await page.getByLabel("Search images", { exact: true }).fill("sunset");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page
    .getByRole("button", { name: "Mark image-1.png irrelevant", exact: true })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Mark image-2.png irrelevant", exact: true })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Load more results", exact: true })
    .click();
  await expect(page.locator(".image-card")).toHaveCount(2);
});
