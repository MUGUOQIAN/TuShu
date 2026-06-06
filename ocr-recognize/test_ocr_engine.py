import base64
import json
import unittest
from unittest.mock import patch

import index
import ocr_engine
from prompt_templates import INVOICE_TEMPLATE, TEMPLATE_MAP


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_prompt(self):
        invoice_fields = TEMPLATE_MAP["invoice"]["fields"]
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "发票号码: 12345678\n"
                        "开票日期: 2026年6月6日\n"
                        "购买方名称: 上海采购有限公司\n"
                        "销售方名称: 上海销售有限公司\n"
                        "价税合计金额: ¥1,234.50\n"
                        "税额: 123.45"
                    ]
                }
            }
        )
        chat_response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "发票号码": "12345678",
                                    "开票日期": "2026-06-06",
                                    "购买方名称": "上海采购有限公司",
                                    "销售方名称": "上海销售有限公司",
                                    "价税合计金额": "1234.50",
                                    "税额": "123.45",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", side_effect=[layout_response, chat_response]
        ) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"small image").decode("utf-8"),
                INVOICE_TEMPLATE,
                expected_fields=invoice_fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])
        chat_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码: 12345678", chat_payload["messages"][0]["content"])

    def test_business_card_nested_layout_data_is_mapped(self):
        business_fields = TEMPLATE_MAP["business_card"]["fields"]
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "上海示例科技有限公司",
                        "张三",
                        "销售经理",
                        "手机 13800138000",
                        "zhangsan@example.com",
                        "地址: 上海市浦东新区示例路 1 号",
                    ]
                }
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=layout_response
        ) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"small image").decode("utf-8"),
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=business_fields,
                template_type="business_card",
            )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海示例科技有限公司")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")


class HandlerTest(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type_to_llm(self):
        body = {
            "image_base64": base64.b64encode(b"small image").decode("utf-8"),
            "template_type": "invoice",
        }
        with patch(
            "index.call_llm",
            return_value={
                "发票号码": "12345678",
                "开票日期": "2026-06-06",
                "购买方名称": "上海采购有限公司",
                "销售方名称": "上海销售有限公司",
                "价税合计金额": "1234.50",
                "税额": "123.45",
            },
        ) as call_llm:
            response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["发票号码"], "12345678")
        call_llm.assert_called_once()
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["expected_fields"], TEMPLATE_MAP["invoice"]["fields"])
        self.assertEqual(kwargs["template_type"], "invoice")


if __name__ == "__main__":
    unittest.main()
