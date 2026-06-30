import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def make_image_base64(fmt="JPEG", size=(32, 32)) -> str:
    image = Image.new("RGB", size, "white")
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_chat_model(self):
        fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        image_base64 = make_image_base64()
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if "layout_parsing" in url:
                return FakeResponse(
                    {
                        "data": {
                            "md_results": [
                                "发票号码: 12345678\n开票日期: 2026年6月30日\n价税合计金额: ¥123.45"
                            ]
                        }
                    }
                )
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_module_dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2026-06-30",
                                        "购买方名称": "测试公司",
                                        "销售方名称": "销售公司",
                                        "价税合计金额": "123.45",
                                        "税额": "11.23",
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                "提取发票字段",
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "123.45")
        self.assertEqual(len(calls), 2)
        self.assertIn("layout_parsing", calls[0][0])
        self.assertIn("chat/completions", calls[1][0])

    def test_nested_business_card_layout_text_is_mapped(self):
        image_base64 = make_image_base64()

        def fake_post(url, headers=None, json=None, timeout=None):
            self.assertIn("layout_parsing", url)
            return FakeResponse(
                {
                    "data": {
                        "md_results": "Nicole Lee\nSales Manager\n13800138000\nnicole@example.com\nExample Co., Ltd"
                    }
                }
            )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                "提取名片字段",
                expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Nicole Lee")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "nicole@example.com")

    def test_small_png_keeps_png_data_uri_mime(self):
        image_base64 = make_image_base64(fmt="PNG")

        compressed, mime_type = ocr_engine._compress_base64_image(image_base64)

        self.assertEqual(mime_type, "image/png")
        self.assertEqual(base64.b64decode(compressed), base64.b64decode(image_base64))

    def test_high_resolution_small_image_is_resized(self):
        image_base64 = make_image_base64(fmt="JPEG", size=(2200, 80))

        compressed, mime_type = ocr_engine._compress_base64_image(image_base64)

        self.assertEqual(mime_type, "image/jpeg")
        with Image.open(io.BytesIO(base64.b64decode(compressed))) as image:
            self.assertLessEqual(max(image.size), 1280)


class HandlerTest(unittest.TestCase):
    def test_handler_passes_template_context_to_llm(self):
        image_base64 = make_image_base64()
        event = {
            "body": json.dumps(
                {"image_base64": image_base64, "template_type": "invoice"},
                ensure_ascii=False,
            )
        }

        with patch(
            "index.call_llm",
            return_value={
                "发票号码": "12345678",
                "开票日期": "",
                "购买方名称": "",
                "销售方名称": "",
                "价税合计金额": "",
                "税额": "",
            },
        ) as call_llm:
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertIn("发票号码", kwargs["expected_fields"])

    def test_handler_rejects_all_empty_result(self):
        image_base64 = make_image_base64()
        event = {
            "body": json.dumps(
                {"image_base64": image_base64, "template_type": "invoice"},
                ensure_ascii=False,
            )
        }

        with patch("index.call_llm", return_value={"md_results": ["只有layout外壳"]}):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 422)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])

    def test_handler_rejects_blank_custom_fields(self):
        image_base64 = make_image_base64()
        event = {
            "body": json.dumps(
                {
                    "image_base64": image_base64,
                    "template_type": "custom",
                    "custom_fields": " , ",
                },
                ensure_ascii=False,
            )
        }

        response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])


def json_module_dumps(value):
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
