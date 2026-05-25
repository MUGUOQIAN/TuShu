import base64
import json
import unittest
from unittest.mock import patch

from ocr_engine import call_llm


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_base64 = base64.b64encode(b"small image").decode("utf-8")

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_layout_wrapper_is_structured_from_ocr_text(self, mock_post):
        layout_payload = {
            "data": {
                "md_results": "发票号码: 12345678\n开票日期: 2026年5月25日",
                "layout_details": [],
            }
        }
        structured_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"发票号码": "12345678", "开票日期": "2026-05-25"},
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        mock_post.side_effect = [
            FakeResponse(200, layout_payload),
            FakeResponse(200, structured_payload),
        ]

        result = call_llm(
            self.image_base64,
            "请提取发票字段",
            expected_fields=["发票号码", "开票日期"],
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-05-25")
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("/layout_parsing", mock_post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", mock_post.call_args_list[1].args[0])

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_business_card_uses_nested_md_results_without_text_model(self, mock_post):
        mock_post.return_value = FakeResponse(
            200,
            {
                "data": {
                    "md_results": "张三\n销售经理\n上海示例有限公司\n手机 13812345678\nzhang@example.com",
                    "layout_details": [],
                }
            },
        )

        result = call_llm(
            self.image_base64,
            "请提取名片字段",
            expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "zhang@example.com")
        self.assertEqual(mock_post.call_count, 1)

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_direct_business_json_short_circuits_when_expected_fields_match(self, mock_post):
        mock_post.return_value = FakeResponse(
            200,
            {
                "data": {
                    "发票号码": "87654321",
                    "开票日期": "2026-05-25",
                }
            },
        )

        result = call_llm(
            self.image_base64,
            "请提取发票字段",
            expected_fields=["发票号码", "开票日期"],
        )

        self.assertEqual(result["发票号码"], "87654321")
        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
