// Benchmark-only rendering: use Chromium's shaping for Indic/RTL/CJK text.
import fs from "node:fs/promises";
import { chromium } from "../../frontend/node_modules/@playwright/test/index.mjs";
const config = JSON.parse(await fs.readFile(process.argv[2], "utf8"));
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1500, height: 600 }, deviceScaleFactor: 1, offline: true, locale: "en-US" });
  const page = await context.newPage();
  for (const row of config.images) {
    const font = await fs.readFile(row.font);
    await page.setContent(`<style>@font-face{font-family:Fixture;src:url(data:font/ttf;base64,${font.toString("base64")})}*{box-sizing:border-box}body{margin:0;background:${row.dark ? "#18201c" : "white"};color:${row.dark ? "white" : "black"}}main{font-family:Fixture;font-size:42px;font-weight:400;font-synthesis:none;white-space:pre;line-height:110px;padding:65px 80px;direction:${row.rtl ? "rtl" : "ltr"}}</style><main></main>`);
    await page.evaluate(async (text) => { await document.fonts.load("42px Fixture"); document.querySelector("main").textContent = text; await document.fonts.ready; }, row.text);
    await page.screenshot({ path: row.output });
  }
  console.log(JSON.stringify({ browser: browser.version(), images: config.images.length }));
  await context.close();
} finally { await browser.close(); }
