import base64
import unittest
from unittest.mock import patch

import ocr_engine


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class OcrEngineTests(unittest.TestCase):
    def setUp(self):
        self._old_api_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"
        self.image_base64 = base64.b64encode(b"tiny").decode("utf-8")

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_api_key

    def test_layout_text_is_extracted_with_template_prompt(self):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse(
                    payload={
                        "md_results": [
                            "发票号码: 123456\n开票日期: 2026年5月6日\n价税合计金额: ¥1,234.56"
                        ]
                    }
                )
            return FakeResponse(
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"发票号码":"123456","开票日期":"2026-05-06","价税合计金额":"1234.56"}'
                            }
                        }
                    ]
                }
            )

        with patch("ocr_engine.requests.post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                self.image_base64,
                "请提取发票号码、开票日期、价税合计金额。",
                ["发票号码", "开票日期", "价税合计金额"],
            )

        self.assertEqual(result["发票号码"], "123456")
        self.assertEqual(result["价税合计金额"], "1234.56")
        self.assertEqual(len(calls), 2)
        chat_payload = calls[1][1]
        user_content = chat_payload["messages"][1]["content"]
        self.assertIn("请提取发票号码、开票日期、价税合计金额。", user_content)
        self.assertIn("价税合计金额: ¥1,234.56", user_content)

    def test_non_business_template_does_not_silently_fallback_to_business_card(self):
        def fake_post(url, headers, json, timeout):
            if url.endswith("/layout_parsing"):
                return FakeResponse(payload={"md_results": ["发票号码: 123456"]})
            return FakeResponse(status_code=500, text="extract failed")

        with patch("ocr_engine.requests.post", side_effect=fake_post):
            with self.assertRaises(RuntimeError):
                ocr_engine.call_llm(
                    self.image_base64,
                    "请提取发票号码。",
                    ["发票号码"],
                )


if __name__ == "__main__":
    unittest.main()
