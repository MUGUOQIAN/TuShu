import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _image_base64(fmt="JPEG", size=(24, 24)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码: 12345678\n开票日期: 2026年7月1日\n价税合计金额: ¥88.00",
                    "购买方名称: 上海测试有限公司\n销售方名称: 北京样例有限公司\n税额: 8.00",
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
                                "开票日期": "2026-07-01",
                                "购买方名称": "上海测试有限公司",
                                "销售方名称": "北京样例有限公司",
                                "价税合计金额": "88.00",
                                "税额": "8.00",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=[FakeResponse(layout_payload), FakeResponse(chat_payload)]
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                INVOICE_TEMPLATE,
                expected_fields=["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "88.00")
        self.assertEqual(post.call_args_list[0].args[0], "https://open.bigmodel.cn/api/paas/v4/layout_parsing")
        self.assertEqual(post.call_args_list[1].args[0], "https://open.bigmodel.cn/api/paas/v4/chat/completions")

    def test_business_card_extracts_nested_layout_text_without_company_substring_false_positive(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "上海玖协机械有限公司\nScott Chen\n销售经理\nM 13800138000\nscott@example.com"
                ]
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", return_value=FakeResponse(layout_payload)
        ):
            result = ocr_engine.call_llm(
                _image_base64(),
                "名片",
                expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Scott Chen")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "scott@example.com")

    def test_small_png_keeps_png_data_uri(self):
        uri = ocr_engine._image_data_uri_from_base64(_image_base64(fmt="PNG"))
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_compact_high_resolution_image_is_resized(self):
        uri = ocr_engine._image_data_uri_from_base64(_image_base64(fmt="PNG", size=(2400, 80)))
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        encoded = uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as img:
            self.assertLessEqual(max(img.size), 1280)


class HandlerTest(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type(self):
        captured = {}

        def fake_call_llm(image_base64, prompt, expected_fields, template_type):
            captured["expected_fields"] = expected_fields
            captured["template_type"] = template_type
            return {"发票号码": "12345678"}

        event = {"body": json.dumps({"image_base64": "abc", "template_type": "invoice"})}
        with patch.object(index, "call_llm", side_effect=fake_call_llm):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(captured["template_type"], "invoice")
        self.assertIn("发票号码", captured["expected_fields"])

    def test_handler_rejects_all_empty_successes(self):
        event = {"body": json.dumps({"image_base64": "abc", "template_type": "invoice"})}
        with patch.object(index, "call_llm", return_value={}):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 422)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])


if __name__ == "__main__":
    unittest.main()
