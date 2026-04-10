
import http.server
import os
from http.server import ThreadingHTTPServer


def _loopback_host() -> str:
    return (os.environ.get("DASHBOARD_LOOPBACK_HOST") or "127.0.0.1").strip() or "127.0.0.1"


def _default_allowed_origins(host: str) -> list[str]:
    aliases = {host}
    if host == "127.0.0.1":
        aliases.add("localhost")
    elif host == "localhost":
        aliases.add("127.0.0.1")

    origins: list[str] = []
    for alias in sorted(aliases):
        origins.append(f"http://{alias}:8501")
        origins.append(f"http://{alias}:3000")
    return origins


def _parse_allowed_origins(raw_value: str | None, host: str) -> set[str]:
    value = raw_value if raw_value is not None else ",".join(_default_allowed_origins(host))
    return {entry.strip() for entry in value.split(",") if entry.strip()}


_LOOPBACK_HOST = _loopback_host()
_ALLOWED_ORIGINS = _parse_allowed_origins(os.environ.get("CORS_ALLOWED_ORIGINS"), _LOOPBACK_HOST)


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _allowed_origin(self) -> str | None:
        origin = self.headers.get("Origin", "").strip()
        if origin in _ALLOWED_ORIGINS:
            return origin
        return None

    def end_headers(self):
        origin = self._allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self):
        origin = self._allowed_origin()
        vary_values: list[str] = []

        self.send_response(204)

        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            vary_values.append("Origin")

        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")

        request_headers = self.headers.get("Access-Control-Request-Headers", "").strip()
        if request_headers:
            self.send_header("Access-Control-Allow-Headers", request_headers)
            vary_values.append("Access-Control-Request-Headers")

        if vary_values:
            self.send_header("Vary", ", ".join(vary_values))

        self.send_header("Content-Length", "0")
        super().end_headers()

if __name__ == '__main__':
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else 'static'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    os.chdir(directory)
    with ThreadingHTTPServer((_LOOPBACK_HOST, port), CORSRequestHandler) as httpd:
        print(f"Serving static files with CORS (HTTP/1.1) at http://{_LOOPBACK_HOST}:{port}/")
        httpd.serve_forever()
