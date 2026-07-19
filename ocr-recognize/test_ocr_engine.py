import base64
import io
import unittest

from PIL import Image

from ocr_engine import _compress_base64_image, _extract_name


class ExtractNameTests(unittest.TestCase):
    def test_company_substrings_inside_common_names_are_not_rejected(self):
        for name in ("Nicole Smith", "Scott Lee", "Marco Polo", "Lincoln Brown"):
            with self.subTest(name=name):
                self.assertEqual(name, _extract_name([name, "Sales Manager"]))

    def test_standalone_company_suffix_is_still_rejected(self):
        self.assertEqual("", _extract_name(["Acme Co., Ltd", "Sales Manager"]))


class CompressImageTests(unittest.TestCase):
    def test_small_png_is_converted_to_jpeg_for_jpeg_data_uri(self):
        source = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(source, format="PNG")

        encoded = _compress_base64_image(
            base64.b64encode(source.getvalue()).decode("ascii")
        )

        with Image.open(io.BytesIO(base64.b64decode(encoded))) as result:
            self.assertEqual("JPEG", result.format)
            self.assertEqual((32, 32), result.size)

    def test_small_bytes_high_resolution_jpeg_is_resized(self):
        source = io.BytesIO()
        Image.new("RGB", (2000, 1600), "white").save(
            source, format="JPEG", quality=20
        )
        self.assertLess(len(source.getvalue()), 300 * 1024)

        encoded = _compress_base64_image(
            base64.b64encode(source.getvalue()).decode("ascii")
        )

        with Image.open(io.BytesIO(base64.b64decode(encoded))) as result:
            self.assertEqual("JPEG", result.format)
            self.assertLessEqual(max(result.size), 1280)


if __name__ == "__main__":
    unittest.main()
