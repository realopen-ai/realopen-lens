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
    def __init__(
        self,
        window,
        *,
        workspace,
        models,
        status,
        theme,
        on_theme,
        actions,
        precision,
        total_memory,
        recommendation,
        on_precision,
        languages,
        setup_status,
        setup_details,
        setup_progress,
    ):
        self.window = window
        self.theme = theme if theme in PALETTES else "light"
        self.on_theme = on_theme
        self.style = ttk.Style(window)
        self.style.theme_use("clam")
        window.geometry("780x820")
        window.minsize(720, 640)
        outer = ttk.Frame(window, style="Lens.TFrame", padding=(32, 26))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer, style="Lens.TFrame")
        header.pack(fill="x")
        self.logo = tk.Canvas(header, width=42, height=42, highlightthickness=0)
        self.logo.pack(side="left", padx=(0, 12))
        brand = ttk.Frame(header, style="Lens.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="Lens", style="Brand.Lens.TLabel").pack(anchor="w")
        ttk.Label(brand, text="B Y   R E A L O P E N   A I", style="Small.Lens.TLabel").pack(
            anchor="w"
        )
        self.theme_button = ttk.Button(
            header, style="Secondary.Lens.TButton", command=self.toggle_theme
        )
        self.theme_button.pack(side="right")

        # Setup details can grow on a device with missing dependencies. Keep them
        # reachable at small window heights instead of clipping controls.
        viewport = ttk.Frame(outer, style="Lens.TFrame")
        viewport.pack(fill="both", expand=True, pady=(12, 0))
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewport, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            viewport, orient="vertical", command=self.canvas.yview, style="Lens.Vertical.TScrollbar"
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.configure(yscrollcommand=self.update_scrollbar)
        outer = ttk.Frame(self.canvas, style="Lens.TFrame")
        content = self.canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind(
            "<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>", lambda event: self.canvas.itemconfigure(content, width=event.width)
        )
        window.bind(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(
                -int(event.delta / 120 if abs(event.delta) >= 120 else event.delta), "units"
            ),
        )
        window.bind("<Button-4>", lambda event: self.canvas.yview_scroll(-3, "units"))
        window.bind("<Button-5>", lambda event: self.canvas.yview_scroll(3, "units"))

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

        setup = ttk.Frame(outer, style="Card.Lens.TFrame", padding=18)
        setup.pack(fill="x", pady=(0, 16))
        self.setup_expanded = False
        self.setup_toggle = ttk.Button(
            setup, text="▸  Device setup", command=self.toggle_setup, style="Setup.Lens.TButton"
        )
        self.setup_toggle.pack(fill="x")
        ttk.Label(
            setup,
            text="Check local models, text recognition and workspace access.",
            style="Muted.Card.Lens.TLabel",
            wraplength=610,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        self.setup_body = ttk.Frame(setup, style="Inset.Lens.TFrame")
        setup = self.setup_body
        ttk.Label(
            setup,
            textvariable=setup_status,
            style="Card.Lens.TLabel",
            wraplength=610,
            justify="left",
        ).pack(anchor="w", pady=(8, 8))
        ttk.Progressbar(
            setup, variable=setup_progress, maximum=100, style="Lens.Horizontal.TProgressbar"
        ).pack(fill="x", pady=(0, 12))
        ttk.Label(
            setup,
            textvariable=setup_details,
            style="Muted.Card.Lens.TLabel",
            wraplength=610,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))
        from realopen_lens.desktop.languages import LanguagePicker

        self.language_picker = LanguagePicker(
            setup, languages, actions.get("ocr_changed", lambda: None)
        )
        self.language_picker.pack(fill="x", pady=(0, 10))
        setup_controls = ttk.Frame(setup, style="Inset.Lens.TFrame")
        setup_controls.pack(fill="x")
        self.check_setup = ttk.Button(
            setup_controls,
            text="Check setup again",
            command=actions["check"],
            style="Secondary.Lens.TButton",
        )
        self.check_setup.pack(side="left")
        ttk.Button(
            setup,
            text="Download models / cancel",
            command=actions["download"],
            style="Primary.Lens.TButton",
        ).pack(anchor="w", pady=(12, 0))
        ttk.Button(
            setup_controls,
            text="OCR help ↗",
            command=actions["ocr_help"],
            style="Secondary.Lens.TButton",
        ).pack(side="left", padx=10)

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
        choices = ttk.Frame(card, style="Inset.Lens.TFrame")
        choices.pack(fill="x", pady=(0, 8))
        ttk.Label(choices, text="SigLIP2", style="Heading.Card.Lens.TLabel").pack(
            side="left", padx=(0, 14)
        )
        for value, label in (("int8", "INT8 · lower memory"), ("fp32", "FP32 · higher quality")):
            ttk.Radiobutton(
                choices,
                text=label,
                value=value,
                variable=precision,
                command=on_precision,
                style="Lens.TRadiobutton",
            ).pack(side="left", padx=(0, 16))
        ram = (
            f"{total_memory / 1024**3:.1f} GiB RAM"
            if total_memory is not None
            else "RAM unavailable"
        )
        ttk.Label(
            card,
            text=f"{ram} · Suggested: {recommendation.upper()}. E5 stays INT8.",
            style="Small.Card.Lens.TLabel",
        ).pack(anchor="w", pady=(0, 12))
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
            text="●  Private by design  ·  No images leave this device",
            style="Small.Lens.TLabel",
        ).pack(side="left")
        ttk.Button(
            footer, text="Logs", command=actions["logs"], style="Secondary.Lens.TButton"
        ).pack(side="right", padx=(8, 0))
        ttk.Button(
            footer, text="Quit Lens", command=actions["quit"], style="Secondary.Lens.TButton"
        ).pack(side="right")
        self.apply_theme()

    def update_scrollbar(self, first, last):
        self.scrollbar.set(first, last)
        if float(last) - float(first) >= 1:
            self.scrollbar.grid_remove()
        else:
            self.scrollbar.grid()

    def set_setup_expanded(self, expanded):
        self.setup_expanded = expanded
        self.setup_toggle.configure(text=("▾" if expanded else "▸") + "  Device setup")
        if expanded:
            self.setup_body.pack(fill="x", pady=(12, 0))
        else:
            self.setup_body.pack_forget()

    def toggle_setup(self):
        self.set_setup_expanded(not self.setup_expanded)

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()
        self.on_theme(self.theme)

    def apply_theme(self):
        p = PALETTES[self.theme]
        self.window.configure(background=p["page"])
        self.canvas.configure(background=p["page"])
        self.style.layout(
            "Lens.Vertical.TScrollbar",
            [
                (
                    "Vertical.Scrollbar.trough",
                    {
                        "sticky": "ns",
                        "children": [
                            ("Vertical.Scrollbar.thumb", {"sticky": "nswe"}),
                        ],
                    },
                ),
            ],
        )
        self.style.configure(
            "Lens.Vertical.TScrollbar",
            background=p["muted"],
            troughcolor=p["page"],
            bordercolor=p["page"],
            lightcolor=p["muted"],
            darkcolor=p["muted"],
            borderwidth=0,
            arrowsize=8,
            width=8,
            relief="flat",
            troughrelief="flat",
        )
        self.style.map(
            "Lens.Vertical.TScrollbar",
            background=[("active", p["accent"]), ("pressed", p["accent"])],
            lightcolor=[("active", p["accent"]), ("pressed", p["accent"])],
            darkcolor=[("active", p["accent"]), ("pressed", p["accent"])],
        )
        self.style.configure(
            "Lens.Horizontal.TProgressbar",
            background=p["accent"],
            troughcolor=p["soft"],
            bordercolor=p["line"],
            lightcolor=p["accent"],
            darkcolor=p["accent"],
        )
        self.style.configure(
            "Lens.TEntry",
            fieldbackground=p["surface"],
            foreground=p["text"],
            bordercolor=p["line"],
            padding=8,
        )
        self.style.configure("Lens.TFrame", background=p["page"])
        self.style.configure("Inset.Lens.TFrame", background=p["surface"])
        self.style.configure(
            "Lens.TRadiobutton",
            background=p["surface"],
            foreground=p["text"],
            font=("Helvetica", 11),
            indicatorbackground=p["surface"],
            indicatorforeground=p["accent"],
            focuscolor=p["accent"],
        )
        self.style.map(
            "Lens.TRadiobutton",
            background=[("active", p["surface"])],
            indicatorbackground=[("selected", p["accent"])],
        )
        self.style.configure(
            "Lens.TCheckbutton",
            background=p["surface"],
            foreground=p["text"],
            font=("Helvetica", 11),
            indicatorbackground=p["surface"],
            indicatorforeground=p["accent"],
            focuscolor=p["accent"],
        )
        self.style.map(
            "Lens.TCheckbutton",
            background=[("active", p["surface"])],
            indicatorbackground=[("selected", p["accent"])],
        )
        self.style.configure(
            "Card.Lens.TFrame",
            background=p["surface"],
            bordercolor=p["line"],
            lightcolor=p["line"],
            darkcolor=p["line"],
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
        self.style.configure(
            "Setup.Lens.TButton",
            background=p["surface"],
            foreground=p["text"],
            font=("Helvetica", 12, "bold"),
            anchor="w",
            padding=0,
            borderwidth=0,
            focuscolor=p["accent"],
        )
        self.style.map("Setup.Lens.TButton", background=[("active", p["surface"])])
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
