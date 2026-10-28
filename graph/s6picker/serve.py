"""Serve the S6 page on 127.0.0.1 only (never 0.0.0.0, never published).

  serve.py [--out DIR] [--port 8766]

Serves exactly OUT/index.html at / and nothing else (no directory listing, no receipt, no cards.json).
"""
import argparse
import http.server
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(pathlib.Path.home() / "catbird-lake-local/s6-picker-20261028"))
ap.add_argument("--port", type=int, default=8766)
A = ap.parse_args()
PAGE = pathlib.Path(A.out).expanduser() / "index.html"


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html"):
            self.send_error(404); return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", A.port), H)
print(f"S6 baseline picker: http://127.0.0.1:{A.port}/  (Ctrl-C to stop)", flush=True)
srv.serve_forever()
