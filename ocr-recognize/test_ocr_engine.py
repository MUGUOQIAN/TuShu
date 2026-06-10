import base64
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTests(unittest.TestCase):
    def setUp(self):
        self.image_base64 = image_base64("JPEG", size=(32, 32))

    @patch.object(ocr_engine, "GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_expected_fields(self, post):
        def fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/layout_parsing"):
                return FakeResponse({
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2026年6月10日\n价税合计金额: ¥1,234.50"
                        ]
                    }
                })
            if url.endswith("/chat/completions"):
                self.assertIn("OCR文本如下", json["messages"][1]["content"])
                self.assertIn("发票号码: 12345678", json["messages"][1]["content"])
                return FakeResponse({
                    "choices": [
                        {
                            "message": {
                                "content": json_module_dumps({
                                    "发票号码": "12345678",
                                    "开票日期": "2026-06-10",
                                    "价税合计金额": "1234.50",
                                })
                            }
                        }
                    ]
                })
            raise AssertionError(f"unexpected url: {url}")

        post.side_effect = fake_post

        result = ocr_engine.call_llm(
            self.image_base64,
            "提取发票字段",
            expected_fields=["发票号码", "开票日期", "价税合计金额"],
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-06-10")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(post.call_count, 2)

    def test_small_png_keeps_png_data_uri(self):
        png_base64 = image_base64("PNG", size=(16, 16))

        data_uri = ocr_engine._prepare_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))
        self.assertEqual(data_uri.rsplit(",", 1)[1], png_base64)

    def test_high_resolution_small_jpeg_is_resized(self):
        jpeg_base64 = image_base64("JPEG", size=(2000, 20))

        data_uri = ocr_engine._prepare_file_data_uri(jpeg_base64)
        encoded = data_uri.rsplit(",", 1)[1]
        resized_bytes = base64.b64decode(encoded)
        with Image.open(io_bytes(resized_bytes)) as resized:
            self.assertLessEqual(max(resized.size), 1280)

    @patch.object(index, "call_llm")
    def test_handler_passes_template_context_to_llm(self, call_llm):
        call_llm.return_value = {"字段A": "值A"}
        event = {
            "body": json.dumps({
                "image_base64": self.image_base64,
                "template_type": "custom",
                "custom_fields": "字段A, 字段B",
            }, ensure_ascii=False)
        }

        response = index.handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"], {"字段A": "值A", "字段B": ""})
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["expected_fields"], ["字段A", "字段B"])
        self.assertEqual(kwargs["template_type"], "custom")


def json_module_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def image_base64(image_format, size=(32, 32)):
    output = io_bytes()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


def io_bytes(value=b""):
    import io

    return io.BytesIO(value)


if __name__ == "__main__":
    unittest.main()
