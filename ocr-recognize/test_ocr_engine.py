import base64
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GLM_API_KEY", "test-key")

from PIL import Image

import ocr_engine
from prompt_templates import TEMPLATE_MAP


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def make_image_base64(image_format="JPEG", size=(80, 40)):
    image = Image.new("RGB", size, color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_wrapper_is_structured_with_chat_model(self):
        invoice_fields = TEMPLATE_MAP["invoice"]["fields"]
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse(
                    {
                        "data": {
                            "md_results": [
                                "发票号码: 12345678\n"
                                "开票日期: 2026年6月21日\n"
                                "价税合计金额: ¥1,234.56"
                            ]
                        }
                    }
                )
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"发票号码":"12345678","开票日期":"2026-06-21",'
                                    '"购买方名称":"","销售方名称":"","价税合计金额":"1234.56","税额":""}'
                                )
                            }
                        }
                    ]
                }
            )

        with patch("ocr_engine.requests.post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                make_image_base64(),
                TEMPLATE_MAP["invoice"]["template"],
                expected_fields=invoice_fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.56")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1]["file"].startswith("data:image/jpeg;base64,"))
        self.assertIn("发票号码: 12345678", calls[1][1]["messages"][1]["content"])

    def test_small_png_keeps_png_mime_type(self):
        png_base64 = make_image_base64("PNG")

        data_uri = ocr_engine._build_glm_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_english_name_with_co_substring_is_not_dropped(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Scott Zhang",
                "Sales Manager",
                "Acme Co., Ltd",
                "13812345678",
                "scott@example.com",
            ]
        )

        self.assertEqual(result["姓名"], "Scott Zhang")


if __name__ == "__main__":
    unittest.main()
