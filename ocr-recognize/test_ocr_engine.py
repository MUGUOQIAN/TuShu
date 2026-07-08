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
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body, ensure_ascii=False)

    def json(self):
        return self._body


def sample_image_base64(format="JPEG") -> str:
    image = Image.new("RGB", (8, 8), color="white")
    output = io.BytesIO()
    image.save(output, format=format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_chat_model(self):
        layout_body = {
            "data": {
                "md_results": [
                    "发票号码 12345678\n开票日期 2026年7月7日\n价税合计金额 ¥12.30"
                ]
            }
        }
        chat_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2026-07-07",
                                "价税合计金额": "12.30",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[FakeResponse(layout_body), FakeResponse(chat_body)],
        ) as post:
            result = ocr_engine.call_llm(
                sample_image_base64(),
                INVOICE_TEMPLATE,
                expected_fields=["发票号码", "开票日期", "价税合计金额"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-07-07")
        self.assertNotIn("md_results", result)
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_nested_layout_text_maps_business_card_and_keeps_scott_name(self):
        layout_body = {
            "data": {
                "md_results": [
                    "Scott Wang",
                    "Sales Manager",
                    "M 13800138000",
                    "scott@example.com",
                ]
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=FakeResponse(layout_body)
        ):
            result = ocr_engine.call_llm(
                sample_image_base64(),
                "extract business card",
                expected_fields=["姓名", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Scott Wang")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "scott@example.com")

    def test_empty_ocr_text_is_not_returned_as_successful_empty_data(self):
        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=FakeResponse({"data": {"md_results": []}})
        ):
            with self.assertRaises(ValueError):
                ocr_engine.call_llm(
                    sample_image_base64(),
                    "extract business card",
                    expected_fields=["姓名"],
                    template_type="business_card",
                )

    def test_small_png_uses_png_data_uri(self):
        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            return_value=FakeResponse({"data": {"md_results": ["张三", "13800138000"]}}),
        ) as post:
            ocr_engine.call_llm(
                sample_image_base64("PNG"),
                "extract business card",
                expected_fields=["姓名"],
                template_type="business_card",
            )

        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["file"].startswith("data:image/png;base64,"))


class IndexHandlerTest(unittest.TestCase):
    def test_handler_passes_template_context_and_rejects_all_empty_fields(self):
        body = {
            "image_base64": "not-used-by-mock",
            "template_type": "invoice",
        }
        event = {"body": json.dumps(body, ensure_ascii=False)}

        with patch("index.call_llm", return_value={"md_results": ["layout shell"]}) as call:
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 422)
        response_body = json.loads(response["body"])
        self.assertFalse(response_body["success"])
        _, kwargs = call.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(kwargs["expected_fields"], ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"])

    def test_custom_template_requires_non_blank_fields(self):
        body = {
            "image_base64": "not-used-by-mock",
            "template_type": "custom",
            "custom_fields": " , ",
        }

        response = index.handler({"body": json.dumps(body, ensure_ascii=False)}, None)

        self.assertEqual(response["statusCode"], 400)
        response_body = json.loads(response["body"])
        self.assertFalse(response_body["success"])


if __name__ == "__main__":
    unittest.main()
