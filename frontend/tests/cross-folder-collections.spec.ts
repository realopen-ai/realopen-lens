import { test, expect } from "@playwright/test";

test("collect library results and retain collection scope for text, similar and upload", async ({
  page,
}) => {
  const folders = ["photos", "screens"].map((name, i) => ({
    name,
    id: String(i + 1).repeat(32),
    root: `/${name}`,
    generation: `g${i}`,
    siglip_precision: i ? "fp32" : "int8",
  }));
  const images = folders.map((f, i) => ({
    rank: i + 1,
    image_id: String(i + 3).repeat(64),
    paths: [`/${f.name}/photo-${i}.png`],
    thumbnail: null,
    width: 64,
    height: 48,
    text: "",
    visual_similarity: 0.8,
    status: "ready",
    issues: [],
    sources: [
      {
        collection: f.name,
        collection_id: f.id,
        generation: f.generation,
        image_id: String(i + 3).repeat(64),
        paths: [`/${f.name}/photo-${i}.png`],
        siglip_precision: f.siglip_precision,
      },
    ],
  }));
  let group: Record<string, any> | null = null,
    task: Record<string, any> = {};
  let members: string[] = [];
  const requests: Record<string, any>[] = [];
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url()),
      path = url.pathname,
      method = route.request().method();
    if (path === "/api/session")
      return route.fulfill({ json: { allowed_roots: ["/photos"] } });
    if (path === "/api/collections") return route.fulfill({ json: folders });
    if (path === "/api/tasks") return route.fulfill({ json: [] });
    if (path === "/api/virtual-collections" && method === "GET")
      return route.fulfill({
        json:
          url.searchParams.get("all_folders") === "true" && group
            ? [
                {
                  ...group,
                  member_count: members.length,
                  available_count: members.length,
                  unavailable_count: 0,
                },
              ]
            : [],
      });
    if (path === "/api/virtual-collections" && method === "POST") {
      const body = route.request().postDataJSON();
      expect(body.all_folders).toBe(true);
      group = {
        id: "a".repeat(32),
        collection_id: "library-v1",
        name: body.name,
        revision: 1,
      };
      return route.fulfill({ status: 201, json: group });
    }
    if (path.endsWith("/members")) {
      const body = route.request().postDataJSON();
      expect(body.all_folders).toBe(true);
      members = method === "DELETE" ? [] : body.image_ids;
      group!.revision += 1;
      return route.fulfill({ json: group });
    }
    if (
      path.startsWith("/api/virtual-collections/") &&
      path.endsWith("/images")
    ) {
      expect(url.searchParams.get("all_folders")).toBe("true");
      return route.fulfill({
        json: {
          results: images.filter((i) => members.includes(i.image_id)),
          total: members.length,
          generation: "library-token",
          unavailable_count: 1,
          scope: group,
        },
      });
    }
    if (path.startsWith("/api/virtual-collections/") && method === "PATCH") {
      const body = route.request().postDataJSON();
      expect(body.all_folders).toBe(true);
      group = { ...group!, name: body.name, revision: group!.revision + 1 };
      return route.fulfill({ json: group });
    }
    if (path.startsWith("/api/tasks/")) {
      if (path.endsWith("/events"))
        return route.fulfill({
          contentType: "text/event-stream",
          body: `event: progress\ndata: ${JSON.stringify({ ...task, status: "completed", revision: 1, result: { results: images, total: 2, ranking: { kind: "visual", mixed_precision_experimental: true }, feedback: { enabled: false } } })}\n\n`,
        });
      const body = path.endsWith("upload")
        ? Object.fromEntries(url.searchParams)
        : route.request().postDataJSON();
      requests.push(body);
      task = {
        id: `t${requests.length}`,
        kind: path.endsWith("search") ? "search" : "similar",
        collection: body.collection,
        all_folders: true,
        virtual_collection: body.virtual_collection,
        virtual_revision: body.virtual_revision,
        status: "queued",
        phase: "queued",
        revision: 0,
        elapsed_seconds: 0,
        progress: {},
        error: null,
        result: null,
      };
      return route.fulfill({ status: 202, json: task });
    }
    if (path.endsWith("/images"))
      return route.fulfill({
        json: { results: [], total: 0, generation: "g0" },
      });
    return route.fulfill({ json: {} });
  });
  await page.goto("/");
  await page.getByLabel("Search in", { exact: true }).selectOption("");
  await page.getByLabel("Search images", { exact: true }).fill("pizza");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.getByLabel("Select photo-0.png", { exact: true }).check();
  await page.getByLabel("Select photo-1.png", { exact: true }).check();
  await page
    .getByRole("button", { name: "New collection", exact: true })
    .click();
  await page.getByLabel("Collection name", { exact: true }).fill("Favorites");
  await page
    .getByRole("button", { name: "Create collection", exact: true })
    .click();
  await page.getByRole("button", { name: "Add selected", exact: true }).click();
  await expect.poll(() => members.length).toBe(2);
  await page
    .getByRole("button", {
      name: "Open virtual collection Favorites",
      exact: true,
    })
    .click();
  await expect(
    page.getByText("1 saved image is unavailable", { exact: false }),
  ).toBeVisible();
  await expect(page.getByLabel("Search in", { exact: true })).toHaveValue("");
  await page.getByLabel("Search images", { exact: true }).fill("cat");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1].virtual_collection).toBe("a".repeat(32));
  expect(requests[1].all_folders).toBe(true);
  await page
    .getByRole("button", { name: "Similar", exact: true })
    .first()
    .click();
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[2].virtual_collection).toBe("a".repeat(32));
  expect(requests[2].all_folders).toBe(true);
  await page.locator('input[type="file"]').setInputFiles({
    name: "query.png",
    mimeType: "image/png",
    buffer: Buffer.from("mock"),
  });
  await expect.poll(() => requests.length).toBe(4);
  expect(requests[3].virtual_collection).toBe("a".repeat(32));
  expect(requests[3].all_folders).toBe("true");
  await page
    .getByRole("button", {
      name: "Manage virtual collection Favorites",
      exact: true,
    })
    .click();
  await page.getByLabel("Collection name", { exact: true }).fill("Best images");
  await page.getByRole("button", { name: "Save name", exact: true }).click();
  await expect(
    page.getByRole("button", {
      name: "Open virtual collection Best images",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByLabel("Select photo-0.png", { exact: true }).check();
  await page
    .getByRole("button", { name: "Remove from collection", exact: true })
    .click();
  await expect.poll(() => members.length).toBe(0);
  await page
    .getByRole("button", { name: "Search all images", exact: true })
    .click();
  await expect(page.getByLabel("Search in", { exact: true })).toHaveValue("");
  await expect(
    page.getByRole("button", { name: "Refresh collection", exact: true }),
  ).toHaveCount(0);
});
