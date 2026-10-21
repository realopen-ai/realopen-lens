import { test, expect, type Page } from "@playwright/test";

async function setup(page: Page) {
  const first = {
    rank: 1,
    image_id: "a".repeat(64),
    paths: ["/photos/red.png"],
    thumbnail: "thumb",
    width: 640,
    height: 480,
    text: "Invoice 004231",
    visual_similarity: 0.8,
    status: "ready",
    issues: [],
  };
  const second = {
    ...first,
    rank: 2,
    image_id: "b".repeat(64),
    paths: ["/photos/green.png"],
  };
  const requests: { url: string; method: string; body: unknown }[] = [];
  let kind = "search";
  let saveRun = false;
  let latest: unknown = null;
  const judgments = new Map<string, string>();
  await page.route("**/api/**", async (route) => {
    const request = route.request(),
      url = new URL(request.url()),
      path = url.pathname;
    requests.push({
      url: path,
      method: request.method(),
      body: request.postData(),
    });
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    if (path === "/api/session") return json({ allowed_roots: ["/photos"] });
    if (path === "/api/collections")
      return json([
        {
          name: "personal",
          root: "/photos",
          generation: "g",
          siglip_precision: "int8",
        },
      ]);
    if (path.endsWith("/images"))
      return json({ results: [first, second], total: 2, generation: "g" });
    if (path.endsWith("/thumbnail") || path.endsWith("/original"))
      return route.fulfill({
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
          "base64",
        ),
      });
    if (path === "/api/tasks") return json(latest ? [latest] : []);
    if (path.startsWith("/api/tasks/") && request.method() === "POST") {
      kind = path.split("/").at(-1)!;
      saveRun = Boolean(JSON.parse(request.postData() || "{}").save_run);
      return json(
        {
          id: "test",
          kind,
          collection: "personal",
          status: "queued",
          phase: "queued",
          progress: {},
          revision: 0,
          elapsed_seconds: 0,
          error: null,
          result: null,
        },
        202,
      );
    }
    if (path.endsWith("/events")) {
      const running = {
        id: "test",
        kind,
        collection: "personal",
        status: "running",
        phase: "ocr",
        elapsed_seconds: 1,
        progress: { unique_images: 2, phase_completed_images: 1 },
        error: null,
        revision: 1,
        result: null,
      };
      const done = {
        ...running,
        status: kind === "index" ? "completed_with_errors" : "completed",
        phase: "finished",
        revision: 2,
        result:
          kind === "index"
            ? { images: 2, failed_files: 1, degraded_files: 1 }
            : { results: [second], run_id: saveRun ? "saved" : null },
      };
      latest = done;
      return route.fulfill({
        contentType: "text/event-stream",
        body: [running, done]
          .map((t) => `event: progress\ndata: ${JSON.stringify(t)}\n\n`)
          .join(""),
      });
    }
    if (path.includes("/feedback/")) {
      const id = path.split("/").at(-1)!;
      if (request.method() === "PUT")
        judgments.set(id, JSON.parse(request.postData()!).label);
      if (request.method() === "DELETE") judgments.delete(id);
      return json({
        judgments: Array.from(judgments, ([image_id, label]) => ({
          image_id,
          label,
        })),
      });
    }
    return json({ detail: "Unexpected endpoint" }, 404);
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "View red.png", exact: true }),
  ).toBeVisible();
  return requests;
}

test("browse, inspect and close details without horizontal overflow", async ({
  page,
}) => {
  await setup(page);
  await page.getByRole("button", { name: "View red.png", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("Invoice 004231")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});

test("search receives SSE completion and supports explicit feedback", async ({
  page,
}) => {
  const requests = await setup(page);
  await page.getByLabel("Save for feedback").check();
  await page
    .getByRole("textbox", { name: "Search images" })
    .fill("green image");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Results for “green image”" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Mark green.png relevant", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Mark green.png relevant", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Mark green.png relevant", exact: true }),
  ).toHaveClass(/chosen/);
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Mark green.png relevant", exact: true }),
  ).toHaveClass(/chosen/);
  await page
    .getByRole("button", { name: "Mark green.png relevant", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Mark green.png relevant", exact: true }),
  ).not.toHaveClass(/chosen/);
  expect(
    requests.some((r) => r.method === "PUT" && r.url.includes("/feedback/")),
  ).toBeTruthy();
  expect(
    requests.some((r) => r.method === "DELETE" && r.url.includes("/feedback/")),
  ).toBeTruthy();
  expect(requests.some((r) => r.url.endsWith("/events"))).toBeTruthy();
});

test("indexing form surfaces partial failures from SSE", async ({ page }) => {
  const requests = await setup(page);
  await page.getByRole("button", { name: "Add folder", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByLabel("Folder path").fill("/photos");
  await page.getByRole("button", { name: "Start indexing" }).click();
  await expect(page.getByLabel("Task progress")).toContainText(
    "1 failed, 1 degraded files",
  );
  expect(
    requests.some((r) => r.url === "/api/tasks/index" && r.method === "POST"),
  ).toBeTruthy();
});

test("find similar submits indexed identity", async ({ page }) => {
  const requests = await setup(page);
  await page
    .getByRole("button", { name: "Similar", exact: true })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Similar to red.png" }),
  ).toBeVisible();
  await expect(page.getByLabel("Task progress")).toContainText("finished");
  const request = requests.find((r) => r.url === "/api/tasks/similar");
  expect(JSON.parse(String(request?.body)).image_id).toBe("a".repeat(64));
});

test("failed search never presents old images as new results", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/tasks/test/events", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body:
        "event: progress\ndata: " +
        JSON.stringify({
          id: "test",
          kind: "search",
          collection: "personal",
          status: "failed",
          phase: "failed",
          elapsed_seconds: 1,
          revision: 1,
          error: "Model unavailable",
          progress: {},
          result: null,
        }) +
        "\n\n",
    }),
  );
  await page.getByRole("textbox", { name: "Search images" }).fill("new query");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Model unavailable");
  await expect(page.locator(".image-card")).toHaveCount(0);
});

test("expired search after server restart asks to rerun instead of showing browse results", async ({
  page,
}) => {
  await setup(page);
  await page.evaluate(() => sessionStorage.setItem("lens-task", "expired"));
  await page.route("**/api/tasks", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "expired",
          kind: "search",
          collection: "personal",
          status: "completed",
          phase: "finished",
          elapsed_seconds: 1,
          revision: 1,
          error: null,
          progress: {},
          result: null,
        },
      ]),
    }),
  );
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("Run the search again");
  await expect(page.locator(".image-card")).toHaveCount(0);
});

test("unsaved results expose feedback and explicitly rerun the original search", async ({
  page,
}) => {
  const requests = await setup(page);
  await page
    .getByRole("textbox", { name: "Search images" })
    .fill("original query");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  const relevant = page.getByRole("button", {
    name: "Mark green.png relevant",
    exact: true,
  });
  await expect(relevant).toBeVisible();
  await expect(relevant).toBeDisabled();
  await expect(page.getByLabel("Search feedback")).toContainText(
    "Want to rate these results?",
  );
  expect(requests.filter((r) => r.url === "/api/tasks/search")).toHaveLength(1);
  await page
    .getByRole("textbox", { name: "Search images" })
    .fill("edited but not submitted");
  await page
    .getByRole("button", { name: "Search again with feedback", exact: true })
    .click();
  await expect(relevant).toBeEnabled();
  await expect(page.getByLabel("Save for feedback")).toBeChecked();
  await expect(page.getByLabel("Search feedback")).toContainText(
    "Mark images relevant or irrelevant",
  );
  const searches = requests.filter((r) => r.url === "/api/tasks/search");
  expect(searches).toHaveLength(2);
  expect(JSON.parse(String(searches[0].body)).save_run).toBe(false);
  expect(JSON.parse(String(searches[1].body))).toMatchObject({
    query: "original query",
    save_run: true,
  });
  await relevant.click();
  await expect(relevant).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});
