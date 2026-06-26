import base64
import json
import unittest
from unittest.mock import patch

import index
from ocr_engine import call_llm


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTests(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        image_base64 = base64.b64encode(b"small-image").decode("utf-8")
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "发票号码: 310123456789\n开票日期: 2026年6月26日\n价税合计金额: ¥88.00"
                    ]
                }
            }
        )
        structure_response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "发票号码": "310123456789",
                                    "开票日期": "2026-06-26",
                                    "价税合计金额": "88.00",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

        with patch("ocr_engine.GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", side_effect=[layout_response, structure_response]
        ) as post:
            result = call_llm(
                image_base64,
                "提取发票字段",
                expected_fields=["发票号码", "开票日期", "价税合计金额"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "310123456789")
        self.assertEqual(result["价税合计金额"], "88.00")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_direct_business_json_short_circuits_without_structure_call(self):
        image_base64 = base64.b64encode(b"small-image").decode("utf-8")
        layout_response = FakeResponse({"data": {"发票号码": "310123456789"}})

        with patch("ocr_engine.GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=layout_response
        ) as post:
            result = call_llm(
                image_base64,
                "提取发票字段",
                expected_fields=["发票号码"],
                template_type="invoice",
            )

        self.assertEqual(result, {"发票号码": "310123456789"})
        self.assertEqual(post.call_count, 1)

    def test_business_card_keeps_local_mapping_from_layout_text(self):
        image_base64 = base64.b64encode(b"small-image").decode("utf-8")
        layout_response = FakeResponse(
            {
                "data": {
                    "layout_details": [
                        [
                            {"content": "张三"},
                            {"content": "上海测试科技有限公司"},
                            {"content": "手机 13800138000"},
                            {"content": "zhangsan@example.com"},
                        ]
                    ]
                }
            }
        )

        with patch("ocr_engine.GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=layout_response
        ) as post:
            result = call_llm(
                image_base64,
                "提取名片字段",
                expected_fields=["姓名", "公司", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")
        self.assertEqual(post.call_count, 1)


class HandlerTests(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": base64.b64encode(b"small-image").decode("utf-8"),
                    "template_type": "invoice",
                }
            )
        }

        with patch(
            "index.call_llm",
            return_value={
                "发票号码": "310123456789",
                "开票日期": "2026-06-26",
                "购买方名称": "购买方",
                "销售方名称": "销售方",
                "价税合计金额": "88.00",
                "税额": "8.00",
            },
        ) as mocked:
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        mocked.assert_called_once()
        _, prompt = mocked.call_args.args
        self.assertIn("增值税发票", prompt)
        self.assertEqual(mocked.call_args.kwargs["template_type"], "invoice")
        self.assertEqual(
            mocked.call_args.kwargs["expected_fields"],
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )

    def test_handler_rejects_all_empty_structured_result(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": base64.b64encode(b"small-image").decode("utf-8"),
                    "template_type": "invoice",
                }
            )
        }

        with patch("index.call_llm", return_value={}):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 422)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])
        self.assertIn("未识别到有效字段", body["error"])


if __name__ == "__main__":
    unittest.main()
