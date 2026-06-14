import base64
import json
import unittest

import index
import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self._old_api_key = ocr_engine.GLM_API_KEY
        self._old_post = ocr_engine.requests.post
        ocr_engine.GLM_API_KEY = "test-key"

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_api_key
        ocr_engine.requests.post = self._old_post

    def test_invoice_layout_text_is_structured_instead_of_returning_empty_fields(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse(
                    {
                        "data": {
                            "md_results": [
                                "发票号码: 12345678\n开票日期: 2026年6月1日\n价税合计金额: ¥123.45"
                            ],
                            "layout_details": [],
                        }
                    }
                )
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2026-06-01",
                                        "购买方名称": "",
                                        "销售方名称": "",
                                        "价税合计金额": "123.45",
                                        "税额": "",
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        ocr_engine.requests.post = fake_post

        result = ocr_engine.call_llm(
            TINY_PNG_BASE64,
            INVOICE_TEMPLATE,
            expected_fields=["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "123.45")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1]["file"].startswith("data:image/png;base64,"))
        self.assertIn("/chat/completions", calls[1][0])

    def test_business_card_nested_layout_maps_name_without_co_substring_false_positive(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            return FakeResponse(
                {
                    "data": {
                        "md_results": [
                            "Nicole Wang\nManager\nnicole@example.com\nShanghai Example Co., Ltd"
                        ]
                    }
                }
            )

        ocr_engine.requests.post = fake_post

        result = ocr_engine.call_llm(
            TINY_PNG_BASE64,
            "extract business card",
            expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "Nicole Wang")
        self.assertEqual(result["邮箱"], "nicole@example.com")
        self.assertEqual(len(calls), 1)


class IndexHandlerTest(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type_to_ocr_engine(self):
        captured = {}
        old_call_llm = index.call_llm

        def fake_call_llm(image_base64, prompt, **kwargs):
            captured.update(kwargs)
            return {"发票号码": "12345678"}

        index.call_llm = fake_call_llm
        try:
            response = index.handler(
                {
                    "body": json.dumps(
                        {"image_base64": base64.b64encode(b"fake").decode(), "template_type": "invoice"},
                        ensure_ascii=False,
                    )
                },
                None,
            )
        finally:
            index.call_llm = old_call_llm

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(captured["template_type"], "invoice")
        self.assertIn("发票号码", captured["expected_fields"])


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
