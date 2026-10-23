import { test, expect } from "@playwright/test";

test("library search preserves source media and feedback scope", async ({
  page,
}) => {
  const folders = [
    {
      id: "a".repeat(32),
      name: "photos",
      root: "/photos",
      generation: "g1",
      siglip_precision: "int8",
    },
    {
      id: "b".repeat(32),
      name: "screens",
      root: "/screens",
      generation: "g2",
      siglip_precision: "fp32",
    },
  ];
  const image = {
    rank: 1,
    image_id: "c".repeat(64),
    paths: ["/screens/pizza.png"],
    thumbnail: null,
    width: 64,
    height: 48,
    text: "",
    visual_similarity: 0.8,
    status: "ready",
    issues: [],
    sources: [
      {
        collection: "screens",
        collection_id: folders[1].id,
        generation: "g2",
        image_id: "d".repeat(64),
        paths: ["/screens/pizza.png"],
        siglip_precision: "fp32",
      },
    ],
  };
  const bodies: Record<string, unknown>[] = [];
  let task: Record<string, unknown>;
  let hidden = false;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/session")
      return route.fulfill({ json: { allowed_roots: ["/photos"] } });
    if (path === "/api/collections") return route.fulfill({ json: folders });
    if (path === "/api/virtual-collections" || path === "/api/tasks")
      return route.fulfill({ json: [] });
    if (path.endsWith("/images"))
      return route.fulfill({
        json: { results: [], total: 0, generation: "g1" },
      });
    if (path === "/api/tasks/search" || path === "/api/tasks/similar") {
      const body = route.request().postDataJSON();
      bodies.push(body);
      const kind = path.endsWith("similar") ? "similar" : "search";
      task = {
        id: kind + "-job",
        kind,
        collection: body.all_folders ? "All folders" : body.collection,
        all_folders: Boolean(body.all_folders),
        status: "queued",
        phase: "queued",
        elapsed_seconds: 0,
        revision: 0,
        error: null,
        progress: {},
        result: null,
      };
      return route.fulfill({ status: 202, json: task });
    }
    if (path.endsWith("/events"))
      return route.fulfill({
        contentType: "text/event-stream",
        body: `event: progress\ndata: ${JSON.stringify({
          ...task,
          status: "completed",
          revision: 1,
          result: {
            results: [image],
            run_id: "run",
            ranking: {
              kind: "hybrid",
              sha256: null,
              mixed_precision_experimental: Boolean(task.all_folders),
            },
            feedback: { enabled: true, applied: false },
          },
        })}\n\n`,
      });
    if (path === "/api/feedback/runs/run")
      return route.fulfill({ json: { judgments: [] } });
    if (path === "/api/library/hidden") {
      hidden = true;
      return route.fulfill({
        json: { results: [], total: 0, generation: null },
      });
    }
    return route.fulfill({ json: {} });
  });
  await page.goto("/");
  if ((page.viewportSize()?.width || 1440) < 600)
    await page.getByLabel("Search in", { exact: true }).selectOption("");
  else
    await page
      .getByRole("button", { name: "Image library", exact: true })
      .click();
  await expect(page.getByLabel("Search in", { exact: true })).toHaveValue("");
  await expect(
    page.getByRole("button", { name: "Search by image", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Search images", { exact: true }).fill("pizza");
  await page.getByLabel("Save for feedback", { exact: true }).check();
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "View pizza.png", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".ranking-caption")).toContainText(
    "mixed precision (experimental)",
  );
  expect(bodies[0].all_folders).toBe(true);
  expect(bodies[0].save_run).toBe(true);
  expect(bodies[0].virtual_collection).toBeUndefined();
  await expect(page.locator(".image-frame img")).toHaveAttribute(
    "src",
    new RegExp(
      `/sources/${"b".repeat(32)}/images/${"d".repeat(64)}/thumbnail\\?generation=g2$`,
    ),
  );
  await expect(
    page.getByLabel("Select pizza.png", { exact: true }),
  ).toBeDisabled();
  await page.getByText("Query feedback enabled", { exact: true }).click();
  await page
    .getByRole("button", { name: "Show hidden images", exact: true })
    .click();
  await expect(
    page.getByText("0 hidden images", { exact: true }),
  ).toBeVisible();
  expect(hidden).toBe(true);
  await page.getByRole("button", { name: "Similar", exact: true }).click();
  await expect(page.getByLabel("Search in", { exact: true })).toHaveValue(
    "screens",
  );
  await expect(
    page.getByRole("button", { name: "View pizza.png", exact: true }),
  ).toBeVisible();
  expect(bodies[1].collection).toBe("screens");
  expect(bodies[1].image_id).toBe("d".repeat(64));
  expect(bodies[1].virtual_collection).toBeUndefined();
  await expect(page.locator(".ranking-caption")).not.toContainText(
    "mixed precision",
  );
  await page.getByLabel("Search in", { exact: true }).selectOption("photos");
  await expect(page.getByLabel("Search in", { exact: true })).toHaveValue(
    "photos",
  );
  await expect(
    page.getByRole("button", { name: "Search by image", exact: true }),
  ).toBeEnabled();
});
