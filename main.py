from __future__ import annotations

import tkinter as tk

from src.config import default_config
from src.gui import SearchApp
from src.search_service import load_search_engine


def main() -> None:
    config = default_config()
    engine = load_search_engine(config)

    root = tk.Tk()
    SearchApp(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()