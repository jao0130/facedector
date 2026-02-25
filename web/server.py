"""
VitalSense — Local development server (static files only).
All inference runs client-side in the browser.

Adds Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
so that SharedArrayBuffer is available (required by ONNX Runtime Web
multi-threaded WASM backend).

Usage:
    cd D:\\Projects\\facedector
    python web/server.py

Open http://localhost:8000
"""

import http.server
from pathlib import Path

STATIC_DIR = str(Path(__file__).resolve().parent / "static")


class COOPHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler with cross-origin isolation headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()


def main():
    port = 8000
    with http.server.HTTPServer(("0.0.0.0", port), COOPHandler) as httpd:
        print(f"[VitalSense] http://localhost:{port}")
        print(f"[VitalSense] Serving from: {STATIC_DIR}")
        print(f"[VitalSense] COOP/COEP headers enabled (SharedArrayBuffer).")
        print(f"[VitalSense] Press Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
