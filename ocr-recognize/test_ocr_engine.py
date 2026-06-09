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


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        ocr_engine.GLM_API_KEY = "test-key"

    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        layout_response = FakeResponse({
            "data": {
                "md_results": (
                    "发票号码 12345678\n"
                    "开票日期 2026年6月9日\n"
                    "购买方名称 测试购买方\n"
                    "销售方名称 测试销售方\n"
                    "价税合计金额 ¥123.45\n"
                    "税额 12.34"
                ),
                "layout_details": [],
            }
        })
        chat_response = FakeResponse({
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "发票号码": "12345678",
                        "开票日期": "2026年6月9日",
                        "购买方名称": "测试购买方",
                        "销售方名称": "测试销售方",
                        "价税合计金额": "¥123.45",
                        "税额": "12.34",
                    }, ensure_ascii=False)
                }
            }]
        })
        body = {
            "image_base64": base64.b64encode(b"small-image").decode("ascii"),
            "template_type": "invoice",
        }

        with patch("ocr_engine.requests.post", side_effect=[layout_response, chat_response]) as post:
            response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(200, response["statusCode"])
        payload = json.loads(response["body"])
        self.assertTrue(payload["success"])
        self.assertEqual({
            "发票号码": "12345678",
            "开票日期": "2026-06-09",
            "购买方名称": "测试购买方",
            "销售方名称": "测试销售方",
            "价税合计金额": "123.45",
            "税额": "12.34",
        }, payload["data"])
        self.assertEqual(2, post.call_count)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])
        chat_prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("发票号码 12345678", chat_prompt)

    def test_custom_template_uses_chat_structuring_after_layout_ocr(self):
        layout_response = FakeResponse({
            "data": {
                "md_results": "项目 云服务费\n金额 88.00",
                "layout_details": [],
            }
        })
        chat_response = FakeResponse({
            "choices": [{
                "message": {
                    "content": '{"项目": "云服务费", "金额": "88.00"}'
                }
            }]
        })
        body = {
            "image_base64": base64.b64encode(b"small-image").decode("ascii"),
            "template_type": "custom",
            "custom_fields": "项目,金额",
        }

        with patch("ocr_engine.requests.post", side_effect=[layout_response, chat_response]):
            response = index.handler({"body": json.dumps(body)}, None)

        payload = json.loads(response["body"])
        self.assertTrue(payload["success"])
        self.assertEqual({"项目": "云服务费", "金额": "88.00"}, payload["data"])

    def test_business_card_maps_nested_md_results_string_without_chat_call(self):
        layout_response = FakeResponse({
            "data": {
                "md_results": (
                    "张三\n"
                    "上海测试有限公司\n"
                    "销售经理\n"
                    "手机 13800138000\n"
                    "zhangsan@example.com\n"
                    "上海市浦东新区测试路1号"
                ),
                "layout_details": [],
            }
        })

        with patch("ocr_engine.requests.post", return_value=layout_response) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"small-image").decode("ascii"),
                "",
                expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertEqual("张三", result["姓名"])
        self.assertEqual("上海测试有限公司", result["公司"])
        self.assertEqual("销售经理", result["职位"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("zhangsan@example.com", result["邮箱"])
        self.assertEqual(1, post.call_count)


if __name__ == "__main__":
    unittest.main()
