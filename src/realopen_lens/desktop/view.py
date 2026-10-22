"""Gallery-inspired launcher presentation, separate from the server lifecycle."""

import tkinter as tk
from tkinter import ttk

PALETTES = {
    "light": {
        "page": "#f7f8f5",
        "surface": "#ffffff",
        "text": "#23332c",
        "muted": "#61725a",
        "line": "#e3e8e0",
        "accent": "#275c43",
        "on_accent": "#ffffff",
        "hover": "#204b37",
        "soft": "#eef3e9",
    },
    "dark": {
        "page": "#151f19",
        "surface": "#1c2b22",
        "text": "#e1e9df",
        "muted": "#a5b69f",
        "line": "#344b3a",
        "accent": "#a4cdb0",
        "on_accent": "#183523",
        "hover": "#b6dbbf",
        "soft": "#263b2c",
    },
}


class LauncherView:
    def __init__(self, window, *, workspace, models, status, ocr, theme, on_theme, actions):
        self.window = window
        self.theme = theme if theme in PALETTES else "light"
        self.on_theme = on_theme
        self.style = ttk.Style(window)
        self.style.theme_use("clam")
        window.geometry("780x720")
        window.minsize(720, 700)
        outer = ttk.Frame(window, style="Lens.TFrame", padding=(32, 26))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer, style="Lens.TFrame")
        header.pack(fill="x")
        self.logo = tk.Canvas(header, width=42, height=42, highlightthickness=0)
        self.logo.pack(side="left", padx=(0, 12))
        brand = ttk.Frame(header, style="Lens.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="Lens", style="Brand.Lens.TLabel").pack(anchor="w")
        ttk.Label(brand, text="B Y   R E A L O P E N", style="Small.Lens.TLabel").pack(anchor="w")
        self.theme_button = ttk.Button(
            header, style="Secondary.Lens.TButton", command=self.toggle_theme
        )
        self.theme_button.pack(side="right")

        ttk.Label(outer, text="YOUR IMAGES. YOUR MACHINE.", style="Small.Lens.TLabel").pack(
            anchor="w", pady=(28, 8)
        )
        ttk.Label(outer, text="Your library, within reach.", style="Title.Lens.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            outer,
            text="Search, explore and organize — everything stays on this device.",
            style="Muted.Lens.TLabel",
        ).pack(anchor="w", pady=(8, 20))
        row = ttk.Frame(outer, style="Lens.TFrame")
        row.pack(fill="x")
        ttk.Button(
            row, text="Open image library  →", command=actions["open"], style="Primary.Lens.TButton"
        ).pack(side="left")
        ttk.Button(
            row, text="Stop gallery", command=actions["stop"], style="Secondary.Lens.TButton"
        ).pack(side="left", padx=12)

        live = ttk.Frame(outer, style="Card.Lens.TFrame", padding=16)
        live.pack(fill="x", pady=(22, 16))
        ttk.Label(live, text="●  LOCAL ACTIVITY", style="Small.Card.Lens.TLabel").pack(anchor="w")
        ttk.Label(
            live, textvariable=status, style="Card.Lens.TLabel", wraplength=660, justify="left"
        ).pack(anchor="w", pady=(8, 0))

        card = ttk.Frame(outer, style="Card.Lens.TFrame", padding=18)
        card.pack(fill="x")
        ttk.Label(card, text="Your workspace", style="Heading.Card.Lens.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            text=str(workspace),
            style="Muted.Card.Lens.TLabel",
            wraplength=660,
            justify="left",
        ).pack(anchor="w", pady=(5, 14))
        ttk.Label(card, text="Local models", style="Heading.Card.Lens.TLabel").pack(anchor="w")
        self.model_label = ttk.Label(
            card, text=str(models), style="Muted.Card.Lens.TLabel", wraplength=660, justify="left"
        )
        self.model_label.pack(anchor="w", pady=(5, 12))
        controls = ttk.Frame(card, style="Inset.Lens.TFrame")
        controls.pack(fill="x")
        self.choose_models = ttk.Button(
            controls,
            text="Change model folder",
            command=actions["models"],
            style="Secondary.Lens.TButton",
        )
        self.choose_models.pack(side="left")
        ttk.Button(
            controls,
            text="Download models / cancel",
            command=actions["download"],
            style="Secondary.Lens.TButton",
        ).pack(side="left", padx=(10, 0))
        ttk.Label(
            card,
            text="Stop the gallery to change model folders. Add image folders in the library.",
            style="Small.Card.Lens.TLabel",
            wraplength=660,
        ).pack(anchor="w", pady=(12, 0))

        footer = ttk.Frame(outer, style="Lens.TFrame")
        footer.pack(fill="x", side="bottom", pady=(16, 0))
        ttk.Label(
            footer,
            text="●  Private by design  ·  " + ("OCR available" if ocr else "OCR not installed"),
            style="Small.Lens.TLabel",
        ).pack(side="left")
        ttk.Button(
            footer, text="Logs", command=actions["logs"], style="Secondary.Lens.TButton"
        ).pack(side="right", padx=(8, 0))
        ttk.Button(
            footer, text="Quit Lens", command=actions["quit"], style="Secondary.Lens.TButton"
        ).pack(side="right")
        self.apply_theme()

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()
        self.on_theme(self.theme)

    def apply_theme(self):
        p = PALETTES[self.theme]
        self.window.configure(background=p["page"])
        self.style.configure("Lens.TFrame", background=p["page"])
        self.style.configure("Inset.Lens.TFrame", background=p["surface"])
        self.style.configure(
            "Card.Lens.TFrame",
            background=p["surface"],
            bordercolor=p["line"],
            borderwidth=1,
            relief="solid",
        )
        for suffix, background in (("Lens.TLabel", p["page"]), ("Card.Lens.TLabel", p["surface"])):
            self.style.configure(
                suffix, background=background, foreground=p["text"], font=("Helvetica", 12)
            )
            self.style.configure("Muted." + suffix, foreground=p["muted"], font=("Helvetica", 11))
            self.style.configure("Small." + suffix, foreground=p["muted"], font=("Helvetica", 10))
            self.style.configure("Heading." + suffix, font=("Helvetica", 12, "bold"))
        self.style.configure("Brand.Lens.TLabel", font=("Helvetica", 24, "bold"))
        self.style.configure("Title.Lens.TLabel", font=("Georgia", 29))
        for name, background, foreground, hover in (
            ("Primary", p["accent"], p["on_accent"], p["hover"]),
            ("Secondary", p["surface"], p["text"], p["soft"]),
        ):
            style = name + ".Lens.TButton"
            self.style.configure(
                style,
                background=background,
                foreground=foreground,
                bordercolor=p["line"],
                lightcolor=background,
                darkcolor=background,
                padding=(14, 10),
                relief="flat",
                font=("Helvetica", 11),
                focuscolor=p["accent"],
            )
            self.style.map(
                style,
                background=[("active", hover), ("disabled", p["soft"])],
                foreground=[("disabled", p["muted"])],
            )
        self.theme_button.configure(text="☾  Dark" if self.theme == "light" else "☀  Light")
        self.logo.configure(background=p["page"])
        self.logo.delete("all")
        self.logo.create_rectangle(1, 1, 41, 41, fill=p["accent"], outline=p["accent"])
        self.logo.create_oval(11, 10, 26, 25, outline=p["on_accent"], width=2)
        self.logo.create_line(24, 24, 32, 32, fill=p["on_accent"], width=2)
