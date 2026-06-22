import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine
from prompt_templates import TEMPLATE_MAP


def _tiny_png_base64():
    image = Image.new("RGB", (4, 4), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self._old_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_key

    def test_invoice_layout_text_is_structured_with_expected_fields(self):
        fields = TEMPLATE_MAP["invoice"]["fields"]
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "json": json, "timeout": timeout})
            if url.endswith("/layout_parsing"):
                return _FakeResponse({
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2026年6月22日\n价税合计: ¥100.00"
                        ]
                    }
                })
            return _FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": json_module_dumps({
                                "发票号码": "12345678",
                                "开票日期": "2026-06-22",
                                "价税合计金额": "100.00",
                            })
                        }
                    }
                ]
            })

        with patch("ocr_engine.requests.post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                _tiny_png_base64(),
                TEMPLATE_MAP["invoice"]["template"],
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-06-22")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["json"]["file"].startswith("data:image/png;base64,"))
        self.assertIn("发票号码", calls[1]["json"]["messages"][1]["content"])

    def test_business_card_uses_nested_layout_text_without_company_substring_false_positive(self):
        fields = TEMPLATE_MAP["business_card"]["fields"]

        def fake_post(url, headers=None, json=None, timeout=None):
            return _FakeResponse({
                "data": {
                    "md_results": "Scott Wong\nSales Manager\nM: 13800138000\nscott@example.com"
                }
            })

        with patch("ocr_engine.requests.post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                _tiny_png_base64(),
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=fields,
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Scott Wong")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "scott@example.com")


class HandlerTest(unittest.TestCase):
    def test_handler_passes_template_context_to_llm(self):
        body = {
            "image_base64": "abcd",
            "template_type": "invoice",
        }

        with patch("index.call_llm", return_value={"发票号码": "12345678"}) as fake_call:
            response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(response["statusCode"], 200)
        _, kwargs = fake_call.call_args
        self.assertEqual(kwargs["expected_fields"], TEMPLATE_MAP["invoice"]["fields"])
        self.assertEqual(kwargs["template_type"], "invoice")

    def test_handler_rejects_all_empty_results(self):
        body = {
            "image_base64": "abcd",
            "template_type": "business_card",
        }

        with patch("index.call_llm", return_value={}):
            response = index.handler({"body": json.dumps(body)}, None)

        payload = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(payload["success"])

    def test_handler_rejects_blank_custom_fields(self):
        body = {
            "image_base64": "abcd",
            "template_type": "custom",
            "custom_fields": " , ",
        }

        response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(response["statusCode"], 400)


def json_module_dumps(value):
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
