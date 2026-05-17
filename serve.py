#!/usr/bin/env python3
"""Static server for the SMIRK web demo.

Sets Cross-Origin-Opener-Policy + Cross-Origin-Embedder-Policy so the page is
cross-origin isolated, which is what unlocks SharedArrayBuffer — required for
onnxruntime-web's multi-threaded WASM backend.

COEP: credentialless (rather than require-corp) so cross-origin CDN scripts
(jsdelivr, unpkg) load without needing CORP headers on their responses.
"""
import http.server
import os
import sys

PORT = 8080
HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + fmt % args + "\n")


def main():
    os.chdir(WEB_DIR)
    addr = ("127.0.0.1", PORT)
    with http.server.ThreadingHTTPServer(addr, Handler) as httpd:
        print(f"SMIRK web demo: http://localhost:{PORT}/  (COOP/COEP enabled)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
