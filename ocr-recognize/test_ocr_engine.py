import base64
import json
import unittest
from unittest.mock import patch

import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self._old_api_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"
        self.image_base64 = base64.b64encode(b"small-test-image").decode("utf-8")

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_api_key

    def test_invoice_layout_text_is_structured_with_expected_fields(self):
        fields = ["发票号码", "开票日期", "价税合计金额"]
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码: 12345678\n开票日期: 2026年5月26日\n价税合计: ¥1,234.56"
                ]
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2026-05-26",
                                "价税合计金额": "1234.56",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch(
            "ocr_engine.requests.post",
            side_effect=[_FakeResponse(layout_payload), _FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                self.image_base64,
                INVOICE_TEMPLATE,
                expected_fields=fields,
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.56")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])
        chat_request = post.call_args_list[1].kwargs["json"]
        self.assertEqual(chat_request["response_format"], {"type": "json_object"})
        self.assertIn("发票号码", chat_request["messages"][1]["content"])
        self.assertIn("12345678", chat_request["messages"][1]["content"])

    def test_business_result_inside_data_is_returned_without_extra_call(self):
        fields = ["发票号码", "开票日期"]
        layout_payload = {
            "data": {
                "发票号码": "87654321",
                "开票日期": "2026-05-26",
            }
        }

        with patch(
            "ocr_engine.requests.post",
            return_value=_FakeResponse(layout_payload),
        ) as post:
            result = ocr_engine.call_llm(
                self.image_base64,
                INVOICE_TEMPLATE,
                expected_fields=fields,
            )

        self.assertEqual(
            result,
            {
                "发票号码": "87654321",
                "开票日期": "2026-05-26",
            },
        )
        self.assertEqual(post.call_count, 1)

    def test_business_card_fields_still_use_local_mapping(self):
        fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]
        layout_payload = {
            "md_results": [
                "张三",
                "销售经理",
                "上海测试有限公司",
                "手机 13800138000",
                "zhangsan@example.com",
                "上海市测试路1号",
            ]
        }

        with patch(
            "ocr_engine.requests.post",
            return_value=_FakeResponse(layout_payload),
        ) as post:
            result = ocr_engine.call_llm(
                self.image_base64,
                "",
                expected_fields=fields,
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
