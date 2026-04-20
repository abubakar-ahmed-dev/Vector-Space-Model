from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .output import format_search_results
from .retrieval import VSMSearchEngine


class SearchApp:
    BG = "#f4f7fb"
    PANEL = "#ffffff"
    ACCENT = "#1546a0"
    ACCENT_DARK = "#0f2f73"
    TEXT = "#132238"
    MUTED = "#5b6b84"
    BORDER = "#d9e2ef"
    RESULT_BG = "#0f172a"
    RESULT_FG = "#e2e8f0"

    def __init__(self, root: tk.Tk, engine: VSMSearchEngine) -> None:
        self.root = root
        self.engine = engine
        self.query_var = tk.StringVar()

        self.root.title("Vector Space Model Search")
        self.root.geometry("1080x760")
        self.root.minsize(920, 680)
        self.root.configure(bg=self.BG)

        self._configure_styles()
        self._build_layout()
        self._show_welcome()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Header.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 22, "bold"))
        style.configure("SubHeader.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("Hint.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Search.TButton", background=self.ACCENT, foreground="white", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        style.map("Search.TButton", background=[("active", self.ACCENT_DARK)])
        style.configure("Clear.TButton", background=self.BG, foreground=self.ACCENT, padding=(14, 10), font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=24)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))

        ttk.Label(header, text="Vector Space Model Search", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Search the Trump's speech collection",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        panel = ttk.Frame(container, style="Panel.TFrame", padding=20)
        panel.pack(fill="x", pady=(0, 18))

        ttk.Label(panel, text="Query", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Enter a free-text query and press Search or Enter.", style="Hint.TLabel").pack(anchor="w", pady=(3, 14))

        entry_row = ttk.Frame(panel, style="Panel.TFrame")
        entry_row.pack(fill="x")

        self.query_entry = ttk.Entry(entry_row, textvariable=self.query_var, font=("Segoe UI", 12))
        self.query_entry.pack(side="left", fill="x", expand=True, ipady=8)
        self.query_entry.bind("<Return>", lambda _event: self.search())

        search_button = ttk.Button(entry_row, text="Search", style="Search.TButton", command=self.search)
        search_button.pack(side="left", padx=(12, 0))

        clear_button = ttk.Button(entry_row, text="Clear", style="Clear.TButton", command=self.clear)
        clear_button.pack(side="left", padx=(8, 0))

        results_panel = ttk.Frame(container, style="Panel.TFrame", padding=20)
        results_panel.pack(fill="both", expand=True)

        ttk.Label(results_panel, text="Results", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(results_panel, text="Output matches the CLI formatting exactly.", style="Hint.TLabel").pack(anchor="w", pady=(3, 12))

        results_frame = ttk.Frame(results_panel, style="Panel.TFrame")
        results_frame.pack(fill="both", expand=True)

        self.results_text = tk.Text(
            results_frame,
            wrap="word",
            bg=self.RESULT_BG,
            fg=self.RESULT_FG,
            insertbackground=self.RESULT_FG,
            relief="flat",
            padx=16,
            pady=16,
            font=("Consolas", 11),
        )
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.results_text.configure(state="disabled")

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(container, textvariable=self.status_var, style="Status.TLabel")
        status_bar.pack(anchor="w", pady=(12, 0))

        self.root.bind("<Control-l>", lambda _event: self.clear())
        self.root.bind("<Escape>", lambda _event: self.query_entry.focus_set())
        self.query_entry.focus_set()

    def _show_welcome(self) -> None:
        message = (
            "Top Relevant Documents (cosine similarity > "
            f"{self.engine.alpha_threshold:.4f}):\n\n"
            "Type a query and press Search to begin.\n\n"
            "Total retrieved: 0 documents\n---"
        )
        self._set_results(message)

    def _set_results(self, text: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def clear(self) -> None:
        self.query_var.set("")
        self._show_welcome()
        self._set_status("Ready.")
        self.query_entry.focus_set()

    def search(self) -> None:
        query = self.query_var.get().strip()
        if not query:
            self._set_status("Enter a query first.")
            self._show_welcome()
            return

        results = self.engine.search(query, top_k=None)
        self._set_results(format_search_results(results, self.engine.alpha_threshold))
        self._set_status(f"Retrieved {len(results)} documents for the current query.")
