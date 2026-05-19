import base64
import json
import unittest
from unittest.mock import Mock, patch

import ocr_engine


BUSINESS_FIELDS = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]
INVOICE_FIELDS = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
IMAGE_BASE64 = base64.b64encode(b"small-test-image").decode("utf-8")


def _mock_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.text = json.dumps(payload, ensure_ascii=False)
    response.json.return_value = payload
    return response


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self._api_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._api_key

    @patch("ocr_engine.requests.post")
    def test_business_card_does_not_short_circuit_on_layout_data_wrapper(self, post):
        post.return_value = _mock_response(
            {
                "data": {
                    "md_results": "张三\n上海测试有限公司\n销售经理\n手机 13800138000\nzhangsan@example.com",
                }
            }
        )

        result = ocr_engine.call_llm(
            IMAGE_BASE64,
            "extract business card",
            expected_fields=BUSINESS_FIELDS,
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")
        self.assertNotIn("md_results", result)

    @patch("ocr_engine.requests.post")
    def test_invoice_uses_ocr_text_for_structured_extraction(self, post):
        post.side_effect = [
            _mock_response(
                {
                    "md_results": "发票号码 12345678\n开票日期 2026年5月19日\n价税合计金额 ¥123.45",
                    "layout_details": [],
                }
            ),
            _mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2026年5月19日",
                                        "价税合计金额": "123.45",
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
            IMAGE_BASE64,
            "extract invoice",
            expected_fields=INVOICE_FIELDS,
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "123.45")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])
        chat_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码 12345678", chat_payload["messages"][1]["content"])

    @patch("ocr_engine.requests.post")
    def test_direct_business_payload_still_returns_without_second_call(self, post):
        post.return_value = _mock_response({"data": {"姓名": "李四", "手机": "13900139000"}})

        result = ocr_engine.call_llm(
            IMAGE_BASE64,
            "extract business card",
            expected_fields=BUSINESS_FIELDS,
            template_type="business_card",
        )

        self.assertEqual(result, {"姓名": "李四", "手机": "13900139000"})
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
