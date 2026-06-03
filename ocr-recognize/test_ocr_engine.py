import base64
import json
import unittest
from unittest import mock

import index
import ocr_engine
from prompt_templates import BUSINESS_CARD_TEMPLATE, INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_expected_fields(self):
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "发票号码: 12345678\n"
                        "开票日期: 2026年6月3日\n"
                        "购买方名称: 上海采购有限公司\n"
                        "销售方名称: 北京销售有限公司\n"
                        "价税合计金额: ¥1,234.56\n"
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
                                    "开票日期": "2026年6月3日",
                                    "购买方名称": "上海采购有限公司",
                                    "销售方名称": "北京销售有限公司",
                                    "价税合计金额": "¥1,234.56",
                                    "税额": "123.45",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post", side_effect=[layout_response, chat_response]
        ) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"fake-image").decode("utf-8"),
                INVOICE_TEMPLATE,
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual("12345678", result["发票号码"])
        self.assertEqual("上海采购有限公司", result["购买方名称"])
        self.assertEqual(2, post.call_count)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])
        chat_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码: 12345678", chat_payload["messages"][1]["content"])

    def test_business_card_layout_text_uses_local_mapping_without_chat_call(self):
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "张三\n"
                        "总经理\n"
                        "上海测试有限公司\n"
                        "手机 13800138000\n"
                        "zhangsan@example.com"
                    ]
                }
            }
        )
        fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post", return_value=layout_response
        ) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"fake-image").decode("utf-8"),
                BUSINESS_CARD_TEMPLATE,
                expected_fields=fields,
                template_type="business_card",
            )

        self.assertEqual("张三", result["姓名"])
        self.assertEqual("上海测试有限公司", result["公司"])
        self.assertEqual("总经理", result["职位"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("zhangsan@example.com", result["邮箱"])
        self.assertEqual(1, post.call_count)

    def test_handler_passes_template_type_and_expected_fields_to_llm(self):
        with mock.patch("index.call_llm") as call_llm:
            call_llm.return_value = {
                "发票号码": "12345678",
                "开票日期": "2026年6月3日",
                "购买方名称": "上海采购有限公司",
                "销售方名称": "北京销售有限公司",
                "价税合计金额": "¥1,234.56",
                "税额": "123.45",
            }

            response = index.handler(
                {
                    "body": json.dumps(
                        {
                            "image_base64": "ZmFrZQ==",
                            "template_type": "invoice",
                        },
                        ensure_ascii=False,
                    )
                },
                None,
            )

        self.assertEqual(200, response["statusCode"])
        body = json.loads(response["body"])
        self.assertTrue(body["success"])
        self.assertEqual("2026-06-03", body["data"]["开票日期"])
        self.assertEqual("1234.56", body["data"]["价税合计金额"])
        call_llm.assert_called_once()
        self.assertEqual("invoice", call_llm.call_args.kwargs["template_type"])
        self.assertEqual(
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
            call_llm.call_args.kwargs["expected_fields"],
        )


if __name__ == "__main__":
    unittest.main()
