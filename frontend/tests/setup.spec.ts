import { test, expect, type Page } from "@playwright/test";

async function device(
  page: Page,
  running = false,
  completion?: Promise<void>,
  preference = "eng+ara",
) {
  let request: Record<string, string> = {};
  const calls: string[] = [];
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    calls.push(path);
    if (path === "/api/workspace")
      return route.fulfill({
        json: {
          directory: "/workspace",
          default_directory: "/workspace",
          models_directory: "/models",
          initialized: false,
          is_default: true,
          workspace_id: null,
          checkout: null,
          build: {
            version: "0.1.0",
            code_sha256: "a".repeat(64),
            gallery_sha256: "b".repeat(64),
            package_path: "/installed/lens",
          },
        },
      });
    if (path === "/api/session")
      return route.fulfill({
        json: {
          allowed_roots: ["/photos"],
          default_siglip_precision: "fp32",
          default_ocr_languages: preference,
          ocr_language_options: {
            eng: "English",
            fra: "French",
            ara: "Arabic",
            jpn: "Japanese",
            deu: "German",
          },
        },
      });
    if (
      path === "/api/collections" ||
      path === "/api/virtual-collections" ||
      path === "/api/tasks"
    )
      return route.fulfill({ json: [] });
    if (path === "/api/tasks/setup") {
      request = route.request().postDataJSON();
      return route.fulfill({
        status: 202,
        json: {
          id: "check",
          kind: "setup",
          collection: "workspace",
          status: "queued",
          phase: "queued",
          progress: {},
          revision: 0,
          elapsed_seconds: 0,
          error: null,
          result: null,
        },
      });
    }
    if (path.endsWith("/events")) {
      if (completion) await completion;
      const task = {
        id: "check",
        kind: "setup",
        collection: "workspace",
        status: running ? "running" : "completed",
        phase: running ? "Verifying local model files" : "finished",
        progress: { step: 1, max_steps: 3 },
        revision: 1,
        elapsed_seconds: 1,
        error: null,
        result: running
          ? null
          : {
              ready: false,
              siglip_precision: request.siglip_precision,
              languages: request.languages,
              source: request.source || null,
              ocr: { installed_languages: ["eng"] },
              ocr_guide:
                "https://tesseract-ocr.github.io/tessdoc/Installation.html",
              checks: [
                {
                  id: "workspace",
                  title: "Workspace",
                  ok: true,
                  detail: "/workspace",
                },
                {
                  id: "models",
                  title: "SigLIP2 FP32 + E5 INT8",
                  ok: false,
                  detail: "Missing or incomplete: vision.onnx",
                  action: "Download models in the desktop launcher.",
                },
                {
                  id: "ocr",
                  title: "Text recognition",
                  ok: false,
                  detail: "Missing OCR language packs: ara",
                  action:
                    "Install requested language packs or select installed languages.",
                },
              ],
            },
      };
      return route.fulfill({
        contentType: "text/event-stream",
        body: `event: progress\ndata: ${JSON.stringify(task)}\n\n`,
      });
    }
    return route.fulfill({ json: {} });
  });
  await page.goto("/");
  return { calls, request: () => request };
}

test("first-run setup explains missing requirements without blocking browsing", async ({
  page,
}, info) => {
  await device(page);
  await page.getByRole("button", { name: "Check device setup" }).click();
  const setup = page
    .getByRole("region", { name: "Device setup" })
    .filter({ visible: true });
  await expect(
    setup.getByRole("button", { name: "Check setup", exact: true }),
  ).toBeHidden();
  await setup.getByText("Device setup", { exact: true }).click();
  await setup.getByRole("button", { name: "Check setup", exact: true }).click();
  await expect(setup).toContainText("Setup needs attention before indexing.");
  await expect(setup).toContainText("Missing OCR language packs: ara");
  await expect(setup).toContainText("Download models in the desktop launcher.");
  await expect(
    setup.getByRole("link", { name: /Tesseract installation/ }),
  ).toHaveAttribute(
    "href",
    "https://tesseract-ocr.github.io/tessdoc/Installation.html",
  );
  await page.screenshot({
    path: `../runs/desktop-onboarding-${info.project.name}.png`,
    fullPage: true,
  });
  await setup
    .getByRole("button", { name: "Use installed languages (eng)" })
    .click();
  await expect(setup).toContainText("Settings changed. Check setup again.");
});

test("folder setup checks use current source, precision and language preferences", async ({
  page,
}) => {
  const state = await device(page);
  await page.getByRole("button", { name: "Add folder", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Add a folder to Lens" });
  await dialog.getByText("Device setup", { exact: true }).click();
  await dialog
    .getByRole("button", { name: "Check setup", exact: true })
    .click();
  await expect(dialog).toContainText("Setup needs attention");
  expect(state.request()).toMatchObject({
    siglip_precision: "fp32",
    languages: "eng+ara",
    source: "/photos",
  });
  expect(state.calls).not.toContain("/api/tasks/index");
  await dialog
    .getByRole("button", { name: "Use installed languages (eng)" })
    .click();
  await expect(dialog.getByLabel("Language codes")).toHaveValue("eng");
});

test("setup verification shows live progress and offers task cancellation", async ({
  page,
}) => {
  const state = await device(page, true);
  await page.getByRole("button", { name: "Check device setup" }).click();
  const setup = page
    .getByRole("region", { name: "Device setup" })
    .filter({ visible: true });
  await setup.getByText("Device setup", { exact: true }).click();
  await setup.getByRole("button", { name: "Check setup", exact: true }).click();
  await expect(setup).toContainText("Verifying local model files");
  const progress = page.getByRole("region", { name: "Task progress" });
  await expect(progress).toContainText("Checking device setup");
  await progress.getByRole("button", { name: "Stop", exact: true }).click();
  expect(state.calls).toContain("/api/tasks/check/cancel");
});

test("missing prerequisites reopen a collapsed setup section and manual collapse still works", async ({
  page,
}) => {
  let complete!: () => void;
  const completion = new Promise<void>((resolve) => {
    complete = resolve;
  });
  await device(page, false, completion);
  await page.getByRole("button", { name: "Check device setup" }).click();
  const setup = page
    .getByRole("region", { name: "Device setup" })
    .filter({ visible: true });
  const summary = setup.getByText("Device setup", { exact: true });
  await summary.click();
  await setup.getByRole("button", { name: "Check setup", exact: true }).click();
  await summary.click();
  await expect(setup.locator("details")).not.toHaveAttribute("open");
  complete();
  await expect(setup.locator("details")).toHaveAttribute("open");
  await expect(
    setup.getByText("Setup needs attention before indexing."),
  ).toBeVisible();
  await summary.click();
  await expect(
    setup.getByRole("button", { name: "Check setup", exact: true }),
  ).toBeHidden();
});

test("OCR mode controls preserve manual choices and submit automatic indexing", async ({
  page,
}, info) => {
  const state = await device(page);
  let indexed: Record<string, unknown> | null = null;
  await page.route("**/api/tasks/index", (route) => {
    indexed = route.request().postDataJSON();
    return route.fulfill({
      status: 202,
      json: {
        id: "index",
        kind: "index",
        status: "completed",
        phase: "finished",
        revision: 1,
        collection: "personal",
        result: {},
        progress: {},
        elapsed_seconds: 1,
        error: null,
      },
    });
  });
  await page.getByRole("button", { name: "Add folder", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Add a folder to Lens" });
  await dialog.getByText("Indexing options", { exact: true }).click();
  await expect(
    dialog.getByRole("radio", { name: "Choose languages", exact: true }),
  ).toBeChecked();
  await dialog.getByText("English, Arabic", { exact: true }).click();
  await dialog.getByRole("checkbox", { name: "Japanese", exact: true }).check();
  await expect(dialog.getByLabel("Language codes")).toHaveValue("eng+ara+jpn");
  await page.screenshot({
    path: `../runs/ocr-manual-${info.project.name}.png`,
    fullPage: true,
  });
  await dialog
    .getByRole("radio", { name: "Automatic script groups", exact: true })
    .check();
  await expect(dialog.getByLabel("Language codes")).toHaveCount(0);
  await dialog.getByText("Device setup", { exact: true }).click();
  await dialog
    .getByRole("button", { name: "Check setup", exact: true })
    .click();
  await expect.poll(() => state.request().languages).toBe("auto");
  await expect(
    dialog.getByRole("button", { name: /Use installed languages/ }),
  ).toHaveCount(0);
  await dialog
    .getByRole("radio", { name: "Choose languages", exact: true })
    .check();
  await expect(dialog.getByLabel("Language codes")).toHaveValue("eng+ara+jpn");
  await dialog
    .getByRole("radio", { name: "Automatic script groups", exact: true })
    .check();
  await page.screenshot({
    path: `../runs/ocr-auto-${info.project.name}.png`,
    fullPage: true,
  });
  await dialog
    .getByRole("button", { name: "Start indexing", exact: true })
    .click();
  await expect.poll(() => indexed?.languages).toBe("auto");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("fresh OCR settings default to manual English French and Arabic", async ({
  page,
}) => {
  await device(page, false, undefined, "");
  await page.getByRole("button", { name: "Add folder", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Add a folder to Lens" });
  await dialog.getByText("Indexing options", { exact: true }).click();
  await expect(
    dialog.getByRole("radio", { name: "Choose languages", exact: true }),
  ).toBeChecked();
  await expect(
    dialog.getByRole("radio", { name: "Automatic script groups", exact: true }),
  ).not.toBeChecked();
  await expect(dialog.getByLabel("Language codes")).toHaveValue("eng+fra+ara");
});
