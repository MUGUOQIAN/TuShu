import base64
import json
import unittest
from unittest.mock import patch

import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_base64 = base64.b64encode(b"small image bytes").decode("utf-8")

    @patch.object(ocr_engine, "GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_chat_model(self, post_mock):
        invoice_fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]

        def fake_post(url, **kwargs):
            if url.endswith("/layout_parsing"):
                return FakeResponse(
                    {
                        "data": {
                            "md_results": [
                                "发票号码: 12345678\n开票日期: 2024年1月2日\n价税合计金额: ¥1,234.50"
                            ]
                        }
                    }
                )
            if url.endswith("/chat/completions"):
                prompt = kwargs["json"]["messages"][1]["content"]
                self.assertIn("发票号码: 12345678", prompt)
                self.assertIn("期望字段：发票号码、开票日期", prompt)
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "发票号码": "12345678",
                                            "开票日期": "2024-01-02",
                                            "购买方名称": "",
                                            "销售方名称": "",
                                            "价税合计金额": "1234.50",
                                            "税额": "",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        post_mock.side_effect = fake_post

        result = ocr_engine.call_llm(
            self.image_base64,
            INVOICE_TEMPLATE,
            expected_fields=invoice_fields,
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(post_mock.call_count, 2)

    @patch.object(ocr_engine, "GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_business_card_ignores_layout_wrapper_and_maps_text(self, post_mock):
        post_mock.return_value = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "上海玖协机械有限公司\n赵美娜 Shermin Zhao\n销售经理\nM 13812345678\nshermin@example.com"
                    ]
                }
            }
        )

        result = ocr_engine.call_llm(
            self.image_base64,
            "business card prompt",
            expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "赵美娜 (Shermin Zhao)")
        self.assertEqual(result["公司"], "上海玖协机械有限公司")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "shermin@example.com")
        self.assertEqual(post_mock.call_count, 1)

    @patch.object(ocr_engine, "GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_direct_business_json_with_expected_fields_is_returned(self, post_mock):
        post_mock.return_value = FakeResponse(
            {
                "data": {
                    "姓名": "张三",
                    "公司": "示例公司",
                    "职位": "经理",
                    "手机": "13800000000",
                    "座机": "",
                    "邮箱": "zhang@example.com",
                    "地址": "",
                }
            }
        )

        result = ocr_engine.call_llm(
            self.image_base64,
            "business card prompt",
            expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "示例公司")
        self.assertEqual(post_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
