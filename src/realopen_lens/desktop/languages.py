"""Manual language selection and optional automatic script routing for the launcher."""

import tkinter as tk
from functools import partial
from tkinter import ttk

from realopen_lens.ocr.languages import LANGUAGES


class LanguagePicker(ttk.Frame):
    def __init__(self, parent, variable, changed):
        super().__init__(parent, style="Inset.Lens.TFrame")
        self.variable = variable
        self.changed = changed
        self.mode = tk.StringVar(value="auto" if variable.get() == "auto" else "manual")
        self.manual = (
            variable.get() if variable.get() and variable.get() != "auto" else "eng+fra+ara"
        )
        self.expanded = False
        self.flags = {code: tk.BooleanVar() for code in LANGUAGES}
        ttk.Label(self, text="OCR languages", style="Small.Card.Lens.TLabel").pack(anchor="w")
        modes = ttk.Frame(self, style="Inset.Lens.TFrame")
        modes.pack(fill="x", pady=(6, 8))
        for value, label in (("manual", "Choose languages"), ("auto", "Automatic script groups")):
            ttk.Radiobutton(
                modes,
                text=label,
                variable=self.mode,
                value=value,
                command=self.select_mode,
                style="Lens.TRadiobutton",
            ).pack(side="left", padx=(0, 18))
        self.help = ttk.Label(self, style="Muted.Card.Lens.TLabel", wraplength=610, justify="left")
        self.help.pack(anchor="w", pady=(0, 8))
        self.manual_body = ttk.Frame(self, style="Inset.Lens.TFrame")
        self.toggle = ttk.Button(
            self.manual_body, command=self.toggle_choices, style="Setup.Lens.TButton"
        )
        self.toggle.pack(fill="x")
        self.choices = ttk.Frame(self.manual_body, style="Inset.Lens.TFrame")
        for i, (code, name) in enumerate(LANGUAGES.items()):
            ttk.Checkbutton(
                self.choices,
                text=name,
                variable=self.flags[code],
                command=partial(self.choose, code),
                style="Lens.TCheckbutton",
            ).grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 16), pady=4)
        ttk.Label(self.manual_body, text="Language codes", style="Small.Card.Lens.TLabel").pack(
            anchor="w", pady=(8, 0)
        )
        entry = ttk.Entry(self.manual_body, textvariable=variable, style="Lens.TEntry")
        entry.pack(fill="x", pady=(4, 8))
        entry.bind("<FocusOut>", lambda event: self.changed())
        self.variable.trace_add("write", self.sync)
        self.sync()

    def sync(self, *args):
        value = self.variable.get()
        automatic = value == "auto"
        self.mode.set("auto" if automatic else "manual")
        if automatic:
            self.manual_body.pack_forget()
            self.help.configure(
                text=(
                    "Detect scripts locally, then select matching packs. Uncertain detection "
                    "uses all installed supported packs; short or mixed text can be missed."
                )
            )
        else:
            self.manual = value
            selected = value.split("+")
            for code, flag in self.flags.items():
                flag.set(code in selected)
            self.toggle.configure(
                text=("▾ " if self.expanded else "▸ ")
                + (
                    ", ".join(LANGUAGES.get(code, code) for code in selected if code)
                    or "Choose at least one language"
                )
            )
            self.help.configure(
                text=(
                    "Enable the languages present in your documents. "
                    "English, French and Arabic are selected by default."
                )
            )
            self.manual_body.pack(fill="x")

    def select_mode(self):
        self.variable.set("auto" if self.mode.get() == "auto" else self.manual or "eng+fra+ara")
        self.changed()

    def choose(self, code):
        selected = [item for item in self.variable.get().split("+") if item]
        selected = (
            list(dict.fromkeys([*selected, code]))
            if self.flags[code].get()
            else [item for item in selected if item != code]
        )
        self.variable.set("+".join(selected))
        self.changed()

    def toggle_choices(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.choices.pack(fill="x", after=self.toggle)
        else:
            self.choices.pack_forget()
        self.sync()
