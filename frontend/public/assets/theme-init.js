// Apply the saved preference before the app loads; storage may be unavailable.
(() => {
  let preference = null;
  try {
    const saved = localStorage.getItem("lens-theme");
    if (saved === "light" || saved === "dark") preference = saved;
  } catch {
    // System appearance remains available in restricted browser contexts.
  }
  const theme =
    preference ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", theme === "dark" ? "#141c18" : "#f7f8f5");
})();
