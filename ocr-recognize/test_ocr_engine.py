import base64
import io
import random
import unittest

from PIL import Image

from ocr_engine import _compress_base64_image


def _jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    random.seed(1)
    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()
    for y in range(0, height, 2):
        for x in range(0, width, 4):
            pixels[x, y] = (
                random.randrange(256),
                random.randrange(256),
                random.randrange(256),
            )
    for y in range(40):
        for x in range(40):
            pixels[x, y] = (255, 0, 0)

    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, exif=exif.tobytes())
    return buffer.getvalue()


def _is_red(pixel: tuple) -> bool:
    red, green, blue = pixel
    return red > 200 and green < 40 and blue < 40


def _compress(raw: bytes) -> Image.Image:
    encoded = _compress_base64_image(base64.b64encode(raw).decode("ascii"))
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


class CompressExifOrientationTests(unittest.TestCase):
    def test_portrait_exif_is_baked_into_pixels_before_reencode(self):
        raw = _jpeg_with_orientation(1280, 720, 6)
        self.assertGreater(len(raw), 300 * 1024)

        image = _compress(raw)

        # Orientation 6 要把 1280x720 的传感器画面转成 720x1280。
        # 左上角红块转到视觉上的右上角；不旋转时红块仍在 (20, 20)。
        self.assertEqual(image.size, (720, 1280))
        self.assertTrue(_is_red(image.getpixel((700, 20))))
        self.assertFalse(_is_red(image.getpixel((20, 20))))
        self.assertNotIn(image.getexif().get(0x0112), (2, 3, 4, 5, 6, 7, 8))

    def test_upright_exif_is_not_rotated(self):
        raw = _jpeg_with_orientation(1280, 720, 1)
        self.assertGreater(len(raw), 300 * 1024)

        image = _compress(raw)

        self.assertEqual(image.size, (1280, 720))
        self.assertTrue(_is_red(image.getpixel((20, 20))))
        self.assertNotIn(image.getexif().get(0x0112), (2, 3, 4, 5, 6, 7, 8))


if __name__ == "__main__":
    unittest.main()
