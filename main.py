"""
Splatt2 — DIY Target Shooting Trainer
Entry point with crash logging.
"""

import sys
import os
import traceback

# Ensure the project directory is on the path
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)


def _crash_log_path() -> str:
    return os.path.join(_BASE, "splatt2_crash.log")


def _write_crash_log(exc_text: str):
    import datetime
    path = _crash_log_path()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Crash at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(exc_text)
            f.write("\n")
        return path
    except Exception:
        return None


def main():
    print("Starting Splatt2...")
    from ui.app import SplattApp
    app = SplattApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_text = traceback.format_exc()
        print(exc_text)
        log_path = _write_crash_log(exc_text)

        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            msg = f"Splatt2 crashed unexpectedly.\n\n{exc_text[:400]}\n\n"
            if log_path:
                msg += f"Full details saved to:\n{log_path}\n\nPlease include this file when reporting the issue."
            messagebox.showerror("Splatt2 — Crash", msg)
            root.destroy()
        except Exception:
            pass

        sys.exit(1)
