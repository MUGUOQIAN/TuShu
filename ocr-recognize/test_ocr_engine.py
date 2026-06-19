import json
import unittest
from unittest.mock import Mock, patch

import index
import ocr_engine
from config import MAX_IMAGE_SIZE
from prompt_templates import TEMPLATE_MAP


SMALL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/6X7qN8AAAAASUVORK5CYII="
)


def mock_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = json.dumps(payload, ensure_ascii=False)
    return response


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.api_key_patch = patch.object(ocr_engine, "GLM_API_KEY", "test-key")
        self.api_key_patch.start()

    def tearDown(self):
        self.api_key_patch.stop()

    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_expected_fields(self, post):
        fields = TEMPLATE_MAP["invoice"]["fields"]
        post.side_effect = [
            mock_response({
                "data": {
                    "md_results": (
                        "发票号码：3100261130\n"
                        "开票日期：2026年6月19日\n"
                        "购买方名称：上海测试科技有限公司\n"
                        "销售方名称：北京样例服务有限公司\n"
                        "价税合计：¥123.45\n"
                        "税额：11.23"
                    ),
                    "layout_details": [],
                }
            }),
            mock_response({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "发票号码": "3100261130",
                            "开票日期": "2026-06-19",
                            "购买方名称": "上海测试科技有限公司",
                            "销售方名称": "北京样例服务有限公司",
                            "价税合计金额": "123.45",
                            "税额": "11.23",
                        }, ensure_ascii=False)
                    }
                }]
            }),
        ]

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            TEMPLATE_MAP["invoice"]["template"],
            expected_fields=fields,
            template_type="invoice",
        )

        self.assertEqual("3100261130", result["发票号码"])
        self.assertEqual("123.45", result["价税合计金额"])
        self.assertEqual(2, post.call_count)
        layout_payload = post.call_args_list[0].kwargs["json"]
        self.assertTrue(layout_payload["file"].startswith("data:image/png;base64,"))
        structure_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("OCR文本如下", structure_payload["messages"][1]["content"])
        self.assertIn("发票号码：3100261130", structure_payload["messages"][1]["content"])

    @patch("ocr_engine.requests.post")
    def test_business_card_uses_nested_md_results_string_without_chat_call(self, post):
        post.return_value = mock_response({
            "data": {
                "md_results": "Nicole Wang\nManager\n13800138000\nnicole@example.com",
                "layout_details": [],
            }
        })

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            TEMPLATE_MAP["business_card"]["template"],
            expected_fields=TEMPLATE_MAP["business_card"]["fields"],
            template_type="business_card",
        )

        self.assertEqual("Nicole Wang", result["姓名"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("nicole@example.com", result["邮箱"])
        self.assertEqual(1, post.call_count)


class HandlerTest(unittest.TestCase):
    def test_handler_rejects_oversized_base64_before_calling_llm(self):
        response = index.handler({
            "body": json.dumps({
                "image_base64": "A" * (MAX_IMAGE_SIZE + 1),
                "template_type": "business_card",
            })
        }, None)

        self.assertEqual(413, response["statusCode"])
        self.assertFalse(json.loads(response["body"])["success"])

    @patch("index.call_llm")
    def test_handler_rejects_all_empty_success_result(self, call_llm):
        call_llm.return_value = {}
        response = index.handler({
            "body": json.dumps({
                "image_base64": SMALL_PNG_BASE64,
                "template_type": "invoice",
            })
        }, None)

        body = json.loads(response["body"])
        self.assertEqual(422, response["statusCode"])
        self.assertFalse(body["success"])
        self.assertIn("未识别到有效字段", body["error"])


if __name__ == "__main__":
    unittest.main()
