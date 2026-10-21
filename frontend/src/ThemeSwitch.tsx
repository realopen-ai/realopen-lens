import { useEffect, useRef, useState } from "react";

type Theme = "light" | "dark";
function savedTheme(): Theme | null {
  try {
    const value = localStorage.getItem("lens-theme");
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function ThemeSwitch() {
  const preference = useRef<Theme | null>(savedTheme());
  const [theme, setTheme] = useState<Theme>(
    document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  );
  function apply(value: Theme) {
    document.documentElement.dataset.theme = value;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", value === "dark" ? "#141c18" : "#f7f8f5");
    setTheme(value);
  }
  useEffect(() => {
    const system = matchMedia("(prefers-color-scheme: dark)");
    const update = () =>
      apply(preference.current || (system.matches ? "dark" : "light"));
    const storage = (event: StorageEvent) => {
      if (event.key !== "lens-theme" && event.key !== null) return;
      preference.current = savedTheme();
      update();
    };
    update();
    system.addEventListener("change", update);
    window.addEventListener("storage", storage);
    return () => {
      system.removeEventListener("change", update);
      window.removeEventListener("storage", storage);
    };
  }, []);
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    preference.current = next;
    try {
      localStorage.setItem("lens-theme", next);
    } catch {
      /* Keep the choice for this page session. */
    }
    apply(next);
  }
  const label =
    theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  return (
    <button
      className="theme-switch"
      type="button"
      onClick={toggle}
      aria-label={label}
      title={label}
      aria-pressed={theme === "dark"}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        aria-hidden="true"
      >
        {theme === "dark" ? (
          <path d="M20.7 13A8.7 8.7 0 0 1 11 3.3 8.8 8.8 0 1 0 20.7 13Z" />
        ) : (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
          </>
        )}
      </svg>
      <span>{theme === "dark" ? "Dark" : "Light"}</span>
    </button>
  );
}
