import base64
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("GLM_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ocr_engine  # noqa: E402


class FileDataUriTests(unittest.TestCase):
    def test_small_png_keeps_png_mime_type(self):
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        uri = ocr_engine._build_file_data_uri(base64.b64encode(png_bytes).decode("utf-8"))

        prefix, encoded = uri.split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(encoded), png_bytes)

    def test_small_jpeg_keeps_jpeg_mime_type(self):
        jpeg_bytes = b"\xff\xd8\xff\xe0small-jpeg"
        uri = ocr_engine._build_file_data_uri(base64.b64encode(jpeg_bytes).decode("utf-8"))

        prefix, encoded = uri.split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        self.assertEqual(base64.b64decode(encoded), jpeg_bytes)


if __name__ == "__main__":
    unittest.main()
