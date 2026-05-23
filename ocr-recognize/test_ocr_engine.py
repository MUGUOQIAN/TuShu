import base64
import unittest
from unittest.mock import patch

import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_base64 = base64.b64encode(b"fake-image").decode("utf-8")

    @patch.object(ocr_engine, "GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_prompt(self, mock_post):
        mock_post.side_effect = [
            FakeResponse(
                200,
                {
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2026年5月23日\n价税合计金额: ¥100.00"
                        ]
                    }
                },
            ),
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"发票号码":"12345678","开票日期":"2026-05-23",'
                                    '"购买方名称":"","销售方名称":"","价税合计金额":"100.00","税额":""}'
                                )
                            }
                        }
                    ]
                },
            ),
        ]

        result = ocr_engine.call_llm(
            self.image_base64,
            INVOICE_TEMPLATE,
            expected_fields=["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "100.00")
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("layout_parsing", mock_post.call_args_list[0].args[0])
        self.assertIn("chat/completions", mock_post.call_args_list[1].args[0])
        self.assertIn("发票号码: 12345678", mock_post.call_args_list[1].kwargs["json"]["messages"][1]["content"])

    @patch.object(ocr_engine, "GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_business_card_uses_nested_layout_text(self, mock_post):
        mock_post.return_value = FakeResponse(
            200,
            {
                "data": {
                    "md_results": [
                        "张三",
                        "上海示例科技有限公司",
                        "销售经理",
                        "手机 13800138000",
                    ]
                }
            },
        )

        result = ocr_engine.call_llm(
            self.image_base64,
            "名片抽取",
            expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
