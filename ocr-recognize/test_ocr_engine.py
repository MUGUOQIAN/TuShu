import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_with_chat_model(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if "layout_parsing" in url:
                return FakeResponse({
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2026年7月10日\n价税合计: ¥1,234.50"
                        ]
                    }
                })
            return FakeResponse({
                "choices": [{
                    "message": {
                        "content": '{"发票号码":"12345678","开票日期":"2026-07-10","价税合计金额":"1234.50"}'
                    }
                }]
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                _jpeg_base64(),
                "提取发票号码、开票日期、价税合计金额",
                expected_fields=["发票号码", "开票日期", "价税合计金额"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-07-10")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(len(calls), 2)
        self.assertIn("OCR文本", calls[1][1]["messages"][1]["content"])

    def test_business_card_nested_layout_text_maps_fields(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResponse({
                "data": {
                    "md_results": "Nicole Wang\nSales Manager\nnicole@example.com\n13800138000"
                }
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                _jpeg_base64(),
                "提取名片",
                expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Nicole Wang")
        self.assertEqual(result["职位"], "Sales Manager")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "nicole@example.com")

    def test_small_png_keeps_png_data_uri_mime(self):
        image_base64 = _image_base64("PNG", size=(12, 12))

        upload_base64, mime_type = ocr_engine._prepare_image_for_upload(image_base64)

        self.assertEqual(upload_base64, image_base64)
        self.assertEqual(mime_type, "image/png")

    def test_high_resolution_small_jpeg_is_downscaled(self):
        image_base64 = _image_base64("JPEG", size=(2000, 2000), color=(255, 255, 255))

        upload_base64, mime_type = ocr_engine._prepare_image_for_upload(image_base64)
        decoded = Image.open(io.BytesIO(base64.b64decode(upload_base64)))

        self.assertEqual(mime_type, "image/jpeg")
        self.assertLessEqual(max(decoded.size), 1280)


class HandlerTest(unittest.TestCase):
    def test_handler_passes_template_context_and_rejects_empty_result(self):
        with patch.object(index, "call_llm", return_value={"发票号码": "", "开票日期": ""}) as call_llm:
            response = index.handler({
                "body": json.dumps({
                    "image_base64": "abc",
                    "template_type": "invoice",
                })
            }, None)

        self.assertEqual(response["statusCode"], 422)
        call_llm.assert_called_once()
        self.assertEqual(call_llm.call_args.kwargs["template_type"], "invoice")
        self.assertIn("发票号码", call_llm.call_args.kwargs["expected_fields"])

    def test_handler_defaults_to_business_card(self):
        with patch.object(index, "call_llm", return_value={"姓名": "张三"}):
            response = index.handler({
                "body": json.dumps({
                    "image_base64": "abc",
                })
            }, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["姓名"], "张三")


def _jpeg_base64() -> str:
    return _image_base64("JPEG", size=(16, 16))


def _image_base64(format_name: str, size=(16, 16), color=(200, 40, 40)) -> str:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, format=format_name)
    return base64.b64encode(output.getvalue()).decode("utf-8")


if __name__ == "__main__":
    unittest.main()
