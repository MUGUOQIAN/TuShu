import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def make_image_base64(image_format="JPEG", size=(16, 16)):
    image = Image.new("RGB", size, color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        posts = []
        expected_fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]

        def fake_post(url, headers, json, timeout):
            posts.append({"url": url, "payload": json, "timeout": timeout})
            if "layout_parsing" in url:
                return FakeResponse({
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2024年1月2日\n价税合计金额: ¥1,234.50"
                        ],
                        "layout_details": [],
                    }
                })
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"发票号码":"12345678","开票日期":"2024-01-02",'
                                '"购买方名称":"","销售方名称":"","价税合计金额":"1234.50","税额":""}'
                            )
                        }
                    }
                ]
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                make_image_base64(),
                INVOICE_TEMPLATE,
                expected_fields=expected_fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(len(posts), 2)
        self.assertIn("layout_parsing", posts[0]["url"])
        self.assertIn("chat/completions", posts[1]["url"])
        self.assertIn("发票号码: 12345678", posts[1]["payload"]["messages"][1]["content"])

    def test_business_card_layout_shell_maps_nested_text_fields(self):
        def fake_post(url, headers, json, timeout):
            return FakeResponse({
                "data": {
                    "md_results": "Nicole Smith\nManager\nACME Co.,Ltd\nnicole@example.com\n13812345678",
                    "layout_details": [[{"content": "Address: 1 Road"}]],
                }
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                make_image_base64(),
                "prompt",
                expected_fields=["姓名", "公司", "职位", "手机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Nicole Smith")
        self.assertEqual(result["职位"], "Manager")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "nicole@example.com")

    def test_prepare_file_data_uri_preserves_small_png_mime(self):
        image_base64 = make_image_base64("PNG")

        data_uri = ocr_engine._prepare_file_data_uri(image_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_prepare_file_data_uri_resizes_high_resolution_small_jpeg(self):
        image_base64 = make_image_base64("JPEG", size=(2200, 1200))

        data_uri = ocr_engine._prepare_file_data_uri(image_base64, max_edge=1280)

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertLessEqual(max(image.size), 1280)


if __name__ == "__main__":
    unittest.main()
