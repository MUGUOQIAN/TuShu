import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import index


class LocalHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        query = {}
        if "?" in self.path:
            _, raw_query = self.path.split("?", 1)
            for item in raw_query.split("&"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    query[k] = v
        event = {"queryParameters": query}
        response = index.handler(event, None)
        self._send_json(response.get("statusCode", 500), json.loads(response.get("body", "{}")))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        event = {"body": raw}
        response = index.handler(event, None)
        self._send_json(response.get("statusCode", 500), json.loads(response.get("body", "{}")))


def main():
    server = HTTPServer(("127.0.0.1", 9000), LocalHandler)
    print("Local server started at http://127.0.0.1:9000")
    server.serve_forever()


if __name__ == "__main__":
    main()
