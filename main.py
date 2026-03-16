#!/usr/bin/env python3.10
"""Piano → MIDI Converter — entry point."""
import sys
import os
import socket
import atexit

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


def _register_mdns(ip: str, port: int, name: str = "piano") -> None:
    """Advertise the server as <name>.local via mDNS/Bonjour."""
    try:
        from zeroconf import Zeroconf, ServiceInfo
        info = ServiceInfo(
            "_http._tcp.local.",
            f"Piano to MIDI._http._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={"path": "/"},
            server=f"{name}.local.",
        )
        zc = Zeroconf()
        zc.register_service(info)
        atexit.register(lambda: (zc.unregister_service(info), zc.close()))
        print(f"mDNS:    http://{name}.local:{port}")
    except Exception as e:
        print(f"mDNS registration skipped: {e}")


if __name__ == "__main__":
    _check_deps()
    from server import app
    import webbrowser

    port = 5050
    local_ip = socket.gethostbyname(socket.gethostname())

    print(f"Local:   http://localhost:{port}")
    print(f"Network: http://{local_ip}:{port}")
    _register_mdns(local_ip, port)

    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
