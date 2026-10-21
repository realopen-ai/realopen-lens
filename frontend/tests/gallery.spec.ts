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
    if (path === "/api/virtual-collections") return json([]);
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

test("learning requires explicit activation and can restore hybrid", async ({
  page,
}) => {
  await setup(page);
  const version = "c".repeat(32);
  let active: string | null = null;
  const activations: unknown[] = [];
  await page.route("**/api/training/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/activation")) {
      const body = JSON.parse(route.request().postData() || "{}");
      activations.push(body);
      if (body.model_version && !body.acknowledge_experimental)
        return route.fulfill({
          status: 400,
          contentType: "application/json",
          body: JSON.stringify({
            detail: "Explicit experimental acknowledgment required",
          }),
        });
      active = body.model_version;
      return route.fulfill({
        json: { active: active ? { personal: active } : {} },
      });
    }
    return route.fulfill({
      json: {
        active: active ? { personal: active } : {},
        datasets: [],
        versions: [
          {
            id: version,
            status: "completed",
            evaluation: {
              failed: 0,
              independent_holdout: false,
              regressions: [],
              hybrid: { mrr_at_10: 0.5, ndcg_at_10: 0.5 },
              logistic: { mrr_at_10: 0.6, ndcg_at_10: 0.6 },
            },
          },
        ],
      },
    });
  });
  await page.goto("/");
  await expect(page.getByText("2 images in this collection")).toBeVisible();
  await page.getByText("Learning & models", { exact: true }).click();
  await expect(
    page
      .locator(".training-panel article")
      .getByText("Diagnostic judgments", { exact: false }),
  ).toBeVisible();
  expect(activations).toHaveLength(0);
  await page
    .getByRole("button", { name: "Activate cccccccc", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText("acknowledgment");
  await page
    .getByLabel("Allow experimental activation", { exact: false })
    .check();
  await page
    .getByRole("button", { name: "Activate cccccccc", exact: true })
    .click();
  await expect(
    page.getByText("Logistic cccccccc", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Restore hybrid ranking", exact: true })
    .click();
  await expect(
    page.locator(".training-panel strong").filter({ hasText: /^Hybrid$/ }),
  ).toBeVisible();
  expect(activations).toHaveLength(3);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("theme follows the system until a persistent choice is made", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await setup(page);
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(
    page.getByRole("button", { name: "Switch to light mode" }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(await page.evaluate(() => localStorage.getItem("lens-theme"))).toBe(
    "dark",
  );
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByText("Learning & models", { exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Download review template" }),
  ).toBeVisible();
  const palette = await page.evaluate(() => ({
    page: getComputedStyle(document.documentElement).backgroundColor,
    panel: getComputedStyle(document.querySelector(".training-panel")!)
      .backgroundColor,
    scheme: getComputedStyle(document.documentElement).colorScheme,
  }));
  expect(palette.scheme).toBe("dark");
  expect(palette.page).toBe("rgb(20, 28, 24)");
  expect(palette.panel).toBe("rgb(27, 38, 32)");
  await page.getByRole("button", { name: "Switch to light mode" }).click();
  await page.emulateMedia({ colorScheme: "dark" });
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("theme switch works when browser storage is unavailable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const get = Storage.prototype.getItem;
    const set = Storage.prototype.setItem;
    Object.defineProperty(Storage.prototype, "getItem", {
      value(key: string) {
        if (this === localStorage)
          throw new DOMException("Unavailable", "SecurityError");
        return get.call(this, key);
      },
    });
    Object.defineProperty(Storage.prototype, "setItem", {
      value(key: string, value: string) {
        if (this === localStorage)
          throw new DOMException("Unavailable", "SecurityError");
        return set.call(this, key, value);
      },
    });
  });
  await page.emulateMedia({ colorScheme: "light" });
  await setup(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("virtual collections create, add, browse, rename, remove and delete", async ({
  page,
}) => {
  await setup(page);
  let group: {
    id: string;
    name: string;
    revision: number;
    collection_id: string;
  } | null = null;
  const members = new Set<string>();
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
  await page.route("**/api/virtual-collections**", async (route) => {
    const request = route.request(),
      url = new URL(request.url()),
      body = JSON.parse(request.postData() || "{}");
    const value = () =>
      group
        ? {
            ...group,
            member_count: members.size,
            available_count: members.size,
            unavailable_count: 0,
          }
        : null;
    if (url.pathname === "/api/virtual-collections") {
      if (request.method() === "POST")
        group = {
          id: "c".repeat(32),
          name: body.name,
          revision: 1,
          collection_id: "parent",
        };
      return route.fulfill({
        json: request.method() === "POST" ? value() : group ? [value()] : [],
      });
    }
    if (url.pathname.endsWith("/images"))
      return route.fulfill({
        json: {
          results: members.size ? [first] : [],
          total: members.size,
          generation: "g",
          member_count: members.size,
          unavailable_count: 0,
        },
      });
    if (url.pathname.endsWith("/members")) {
      for (const id of body.image_ids) {
        if (request.method() === "POST") members.add(id);
        else members.delete(id);
      }
      group!.revision++;
    } else if (request.method() === "PATCH") {
      group!.name = body.name;
      group!.revision++;
    } else if (request.method() === "DELETE") group = null;
    return route.fulfill({ json: value() || { deleted: true } });
  });
  await page
    .getByRole("checkbox", { name: "Select red.png", exact: true })
    .check();
  await page
    .getByRole("button", { name: "New collection", exact: true })
    .click();
  await expect(page.locator(".overlay")).toHaveCSS("position", "fixed");
  const dialogBounds = await page.getByRole("dialog").boundingBox();
  const viewport = page.viewportSize()!;
  expect(dialogBounds!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBounds!.y).toBeGreaterThanOrEqual(0);
  expect(dialogBounds!.x + dialogBounds!.width).toBeLessThanOrEqual(
    viewport.width,
  );
  expect(dialogBounds!.y + dialogBounds!.height).toBeLessThanOrEqual(
    viewport.height,
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await page.getByLabel("Collection name", { exact: true }).fill("Work");
  await page
    .getByRole("button", { name: "Create collection", exact: true })
    .click();
  await page.getByRole("button", { name: "Add selected", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Saved the selection");
  await page
    .getByRole("button", { name: "Open virtual collection Work", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "View green.png", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("checkbox", { name: "Select red.png", exact: true })
    .check();
  await page
    .getByRole("button", { name: "Remove from collection", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "View red.png", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", {
      name: "Manage virtual collection Work",
      exact: true,
    })
    .click();
  await page.getByLabel("Collection name", { exact: true }).fill("Favorites");
  await page.getByRole("button", { name: "Save name", exact: true }).click();
  await page
    .getByRole("button", {
      name: "Manage virtual collection Favorites",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Delete collection…", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText(
    "Original image files stay",
  );
  await page
    .getByRole("button", { name: "Delete collection", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "View green.png", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Open virtual collection Favorites",
      exact: true,
    }),
  ).toHaveCount(0);
  await page
    .getByRole("checkbox", { name: "Select red.png", exact: true })
    .check();
  await expect(
    page.getByRole("button", { name: "Add selected", exact: true }),
  ).toBeDisabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("virtual search requests preserve scope and stale membership is explicit", async ({
  page,
}) => {
  await setup(page);
  const group = {
    id: "c".repeat(32),
    name: "I".repeat(80),
    revision: 2,
    collection_id: "parent",
    member_count: 1,
    available_count: 1,
    unavailable_count: 0,
  };
  await page.route("**/api/virtual-collections**", async (route) => {
    if (new URL(route.request().url()).pathname.endsWith("/images")) {
      return route.fulfill({
        json: {
          results: [
            {
              rank: 1,
              image_id: "a".repeat(64),
              paths: ["/photos/red.png"],
              thumbnail: "thumb",
              width: 640,
              height: 480,
              text: "Invoice",
              visual_similarity: 0.8,
              status: "ready",
              issues: [],
            },
          ],
          total: 1,
          generation: "g",
          unavailable_count: 0,
        },
      });
    }
    if (route.request().method() === "POST") {
      group.revision++;
      return route.fulfill({
        status: 409,
        json: { detail: "Virtual collection changed; refresh and retry" },
      });
    }
    return route.fulfill({ json: [group] });
  });
  await page.route("**/api/tasks/test/events", async (route) => {
    const state = {
      id: "test",
      kind: "search",
      collection: "personal",
      virtual_collection: group.id,
      virtual_revision: group.revision,
      status: "completed",
      phase: "finished",
      revision: 2,
      elapsed_seconds: 1,
      error: null,
      progress: {},
      result: {
        run_id: null,
        results: [
          {
            rank: 1,
            image_id: "a".repeat(64),
            paths: ["/photos/red.png"],
            thumbnail: "thumb",
            width: 640,
            height: 480,
            text: "Invoice",
            visual_similarity: 0.8,
            status: "ready",
            issues: [],
          },
        ],
      },
    };
    return route.fulfill({
      contentType: "text/event-stream",
      body: `event: progress\ndata: ${JSON.stringify(state)}\n\n`,
    });
  });
  await page.reload();
  await page
    .getByRole("button", {
      name: `Open virtual collection ${group.name}`,
      exact: true,
    })
    .click();
  const requestPromise = page.waitForRequest(
    (request) =>
      request.url().endsWith("/api/tasks/search") &&
      request.method() === "POST",
  );
  await page
    .getByRole("textbox", { name: "Search images", exact: true })
    .fill("invoice");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  const request = await requestPromise;
  expect(request.postDataJSON()).toMatchObject({
    virtual_collection: group.id,
    virtual_revision: 2,
  });
  await expect(
    page.getByRole("button", { name: "View red.png", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("checkbox", { name: "Select red.png", exact: true })
    .check();
  await page
    .getByLabel("Add to virtual collection", { exact: true })
    .selectOption(group.id);
  await page.getByRole("button", { name: "Add selected", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("refresh and retry");
  await page
    .getByRole("button", { name: "Refresh collection", exact: true })
    .click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  const nextRequest = page.waitForRequest(
    (request) =>
      request.url().endsWith("/api/tasks/search") &&
      request.method() === "POST",
  );
  await page.getByRole("button", { name: "Search", exact: true }).click();
  expect((await nextRequest).postDataJSON()).toMatchObject({
    virtual_revision: 3,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("duplicate review preserves separate originals and provides scan controls", async ({
  page,
}) => {
  await setup(page);
  let scanned = false;
  let state = "unreviewed";
  const image = (key: string) => ({
    rank: 1,
    image_id: key.repeat(64),
    paths: [`/photos/${key}.png`],
    thumbnail: "thumb",
    width: 640,
    height: 480,
    text: "",
    visual_similarity: null,
    status: "ready",
    issues: [],
  });
  await page.route("**/api/duplicates**", async (route) => {
    const request = route.request();
    if (request.method() === "POST")
      state = JSON.parse(request.postData()!).state;
    await route.fulfill({
      json:
        request.method() === "POST"
          ? { state }
          : {
              scanned,
              generation: "g",
              total: scanned ? 1 : 0,
              summary: {
                fingerprinted: 2,
                images: 2,
                errors: [],
                unsupported: [],
              },
              groups: scanned
                ? [
                    {
                      id: "pair",
                      kind: "near",
                      state,
                      image_ids: ["a".repeat(64), "b".repeat(64)],
                      results: [image("a"), image("b")],
                      unavailable: 0,
                    },
                  ]
                : [],
            },
    });
  });
  await page.route("**/api/tasks/duplicates", async (route) => {
    scanned = true;
    await route.fulfill({
      json: {
        id: "dups",
        kind: "duplicates",
        collection: "personal",
        status: "completed",
        phase: "finished",
        revision: 1,
        elapsed_seconds: 1,
        error: null,
        progress: {},
        result: {},
      },
    });
  });
  await page.goto("/");
  await page.getByText("Duplicate groups", { exact: true }).click();
  await expect(
    page.getByText("No scan for the current index snapshot.", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Scan duplicates", exact: true })
    .click();
  await expect(
    page.getByText("Possible near-duplicates", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Open original a.png" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Confirm similarity" }).click();
  await expect(page.locator(".duplicate-group .pill")).toHaveText("confirmed");
  await page.getByRole("button", { name: "Dismiss suggestion" }).click();
  await expect(page.locator(".duplicate-group .pill")).toHaveText("dismissed");
  await page.getByRole("button", { name: "Reset review" }).click();
  await expect(page.locator(".duplicate-group .pill")).toHaveText("unreviewed");
  await expect(page.locator(".duplicate-images figure")).toHaveCount(2);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("selecting a collapsed result adds all exact byte identities to a collection", async ({
  page,
}) => {
  await setup(page);
  const group = {
    id: "d".repeat(32),
    name: "Saved",
    collection_id: "parent",
    revision: 1,
    member_count: 0,
    available_count: 0,
    unavailable_count: 0,
  };
  let members: string[] = [];
  await page.route("**/api/virtual-collections**", async (route) => {
    if (route.request().method() === "POST") {
      members = JSON.parse(route.request().postData()!).image_ids;
      await route.fulfill({ json: group });
    } else await route.fulfill({ json: [group] });
  });
  await page.route("**/api/collections/personal/images?**", (route) =>
    route.fulfill({
      json: {
        results: [
          {
            rank: 1,
            image_id: "a".repeat(64),
            duplicate_ids: ["c".repeat(64)],
            paths: ["/photos/red.png", "/photos/red.bmp"],
            width: 640,
            height: 480,
            thumbnail: "thumb",
            text: "",
            visual_similarity: null,
            status: "ready",
            issues: [],
          },
        ],
        total: 1,
        generation: "g",
      },
    }),
  );
  await page.goto("/");
  await page
    .getByRole("checkbox", { name: "Select red.png", exact: true })
    .check();
  await page
    .getByLabel("Add to virtual collection", { exact: true })
    .selectOption(group.id);
  await page.getByRole("button", { name: "Add selected", exact: true }).click();
  await expect.poll(() => members).toEqual(["a".repeat(64), "c".repeat(64)]);
});
