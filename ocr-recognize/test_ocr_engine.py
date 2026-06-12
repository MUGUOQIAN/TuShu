import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def make_image_base64(image_format="JPEG", size=(32, 32)):
    image = Image.new("RGB", size, color="white")
    output = io.BytesIO()
    image.save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineRegressionTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_chat(self):
        image_base64 = make_image_base64()
        expected_fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        chat_result = {
            "发票号码": "INV-20260601",
            "开票日期": "2026-06-01",
            "购买方名称": "甲方公司",
            "销售方名称": "乙方公司",
            "价税合计金额": "123.45",
            "税额": "6.99",
        }
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse({
                    "data": {
                        "md_results": (
                            "发票号码：INV-20260601\n"
                            "开票日期：2026年6月1日\n"
                            "购买方名称：甲方公司\n"
                            "销售方名称：乙方公司\n"
                            "价税合计金额：¥123.45\n"
                            "税额：¥6.99"
                        ),
                        "layout_details": [],
                    }
                })
            return FakeResponse({
                "choices": [
                    {"message": {"content": json_module.dumps(chat_result, ensure_ascii=False)}}
                ]
            })

        json_module = json
        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                "请提取发票字段",
                expected_fields=expected_fields,
                template_type="invoice",
            )

        self.assertEqual(result, chat_result)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][0].endswith("/layout_parsing"))
        self.assertTrue(calls[1][0].endswith("/chat/completions"))
        self.assertIn("OCR文本", calls[1][1]["messages"][1]["content"])

    def test_business_card_nested_layout_container_is_mapped(self):
        image_base64 = make_image_base64()

        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResponse({
                "data": {
                    "md_results": "Nicole Wang\nManager\nACME Co., Ltd\nnicole@example.com\n13812345678",
                    "layout_details": [],
                }
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                "请提取名片字段",
                expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Nicole Wang")
        self.assertEqual(result["公司"], "ACME Co., Ltd")
        self.assertEqual(result["职位"], "Manager")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "nicole@example.com")

    def test_small_png_uses_png_data_uri(self):
        png_base64 = make_image_base64("PNG")

        data_uri = ocr_engine._prepare_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))
        self.assertEqual(data_uri.split(",", 1)[1], png_base64)

    def test_low_byte_high_resolution_image_is_resized(self):
        png_base64 = make_image_base64("PNG", size=(2400, 80))

        data_uri = ocr_engine._prepare_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        resized_bytes = base64.b64decode(data_uri.split(",", 1)[1])
        with Image.open(io.BytesIO(resized_bytes)) as resized:
            self.assertLessEqual(max(resized.size), 1280)


class HandlerRegressionTest(unittest.TestCase):
    def test_handler_passes_template_context_to_llm(self):
        body = {
            "image_base64": "dummy",
            "template_type": "invoice",
        }

        with patch.object(index, "call_llm", return_value={"发票号码": "INV-1"}) as mock_call:
            response = index.handler({"body": json.dumps(body, ensure_ascii=False)}, None)

        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertTrue(payload["success"])
        _, kwargs = mock_call.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(
            kwargs["expected_fields"],
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )


if __name__ == "__main__":
    unittest.main()
