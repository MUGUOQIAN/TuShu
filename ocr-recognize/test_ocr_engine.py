import base64
import json
import unittest
from unittest.mock import Mock, patch

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
    def test_invoice_layout_text_is_structured_with_expected_fields(self):
        fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        layout_response = FakeResponse(
            {
                "md_results": "发票号码 12345678\n开票日期 2026年5月20日\n价税合计金额 ¥1,234.56",
                "layout_details": [],
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
                                    "开票日期": "2026-05-20",
                                    "购买方名称": "",
                                    "销售方名称": "",
                                    "价税合计金额": "1234.56",
                                    "税额": "",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine, "_compress_base64_image", return_value="compressed"
        ), patch.object(ocr_engine.requests, "post", side_effect=[layout_response, chat_response]) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"image").decode("ascii"),
                "请抽取发票字段",
                expected_fields=fields,
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.56")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])
        self.assertIn("发票号码", post.call_args_list[1].kwargs["json"]["messages"][1]["content"])

    def test_business_card_still_uses_local_mapping_without_chat_call(self):
        fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]
        layout_response = FakeResponse(
            {
                "layout_details": [
                    [
                        {"content": "张三"},
                        {"content": "上海测试有限公司"},
                        {"content": "销售经理"},
                        {"content": "13800138000"},
                        {"content": "zhangsan@example.com"},
                    ]
                ]
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine, "_compress_base64_image", return_value="compressed"
        ), patch.object(ocr_engine.requests, "post", return_value=layout_response) as post:
            result = ocr_engine.call_llm(
                base64.b64encode(b"image").decode("ascii"),
                "请抽取名片字段",
                expected_fields=fields,
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(post.call_count, 1)

    def test_handler_passes_expected_fields_to_ocr_engine(self):
        expected = {"发票号码": "12345678", "开票日期": "2026-05-20"}
        event = {
            "body": json.dumps(
                {
                    "image_base64": base64.b64encode(b"image").decode("ascii"),
                    "template_type": "invoice",
                }
            )
        }

        with patch.object(index, "call_llm", return_value=expected) as call_llm:
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        call_llm.assert_called_once()
        self.assertEqual(
            call_llm.call_args.kwargs["expected_fields"],
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )
        body = json.loads(response["body"])
        self.assertEqual(body["data"]["发票号码"], "12345678")


if __name__ == "__main__":
    unittest.main()
