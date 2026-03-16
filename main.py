#!/usr/bin/env python3.10
"""Piano → MIDI Converter — entry point."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check_deps():
    missing = []
    for pkg, import_name in [
        ("librosa", "librosa"),
        ("mido", "mido"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("flask", "flask"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing dependencies. Install them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    _check_deps()
    from server import app
    import webbrowser
    port = 5050
    print(f"Opening http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    app.run(port=port, debug=False)
