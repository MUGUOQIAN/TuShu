import base64
import json
import unittest
from unittest.mock import patch

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
        self._old_api_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_api_key

    @patch("ocr_engine.requests.post")
    def test_non_card_layout_shell_is_structured_from_ocr_text(self, mock_post):
        mock_post.side_effect = [
            FakeResponse(
                {
                    "data": {
                        "md_results": [
                            "发票号码 12345678\n开票日期 2026年7月3日",
                            "价税合计金额 ¥1,234.50",
                        ]
                    }
                }
            ),
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2026-07-03",
                                        "价税合计金额": "1234.50",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            ),
        ]

        result = ocr_engine.call_llm(
            base64.b64encode(b"small image").decode("utf-8"),
            "提取发票字段",
            expected_fields=["发票号码", "开票日期", "价税合计金额"],
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("/layout_parsing", mock_post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", mock_post.call_args_list[1].args[0])
        self.assertIn("发票号码 12345678", mock_post.call_args_list[1].kwargs["json"]["messages"][0]["content"])

    @patch("ocr_engine.requests.post")
    def test_business_card_uses_nested_layout_text_without_name_substring_false_positive(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "data": {
                    "md_results": "Marco Silva\nSales Manager\n13812345678\nmarco@example.com",
                    "layout_details": [[{"content": "Acme Corporation"}]],
                }
            }
        )

        result = ocr_engine.call_llm(
            base64.b64encode(b"small image").decode("utf-8"),
            "提取名片字段",
            expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "Marco Silva")
        self.assertEqual(result["职位"], "Sales Manager")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "marco@example.com")
        self.assertEqual(mock_post.call_count, 1)

    def test_handler_rejects_all_empty_result(self):
        body = {
            "image_base64": base64.b64encode(b"small image").decode("utf-8"),
            "template_type": "business_card",
        }
        with patch("index.call_llm", return_value={}):
            response = index.handler({"body": json.dumps(body)}, None)

        payload = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(payload["success"])


if __name__ == "__main__":
    unittest.main()
