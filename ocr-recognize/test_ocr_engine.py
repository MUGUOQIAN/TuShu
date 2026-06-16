import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload, ensure_ascii=False)

    def json(self):
        return self._payload


def make_image_base64(image_format="JPEG", size=(20, 20), quality=85):
    output = io.BytesIO()
    Image.new("RGB", size, color="white").save(output, format=image_format, quality=quality)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        ocr_engine.GLM_API_KEY = "test-key"

    @patch("ocr_engine.requests.post")
    def test_invoice_layout_shell_is_structured_instead_of_returned_as_business_json(self, post):
        post.side_effect = [
            FakeResponse(
                payload={
                    "data": {
                        "md_results": [
                            "发票号码 12345678\n开票日期 2024年1月2日\n价税合计金额 ¥1,234.56"
                        ]
                    }
                }
            ),
            FakeResponse(
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2024-01-02",
                                        "价税合计金额": "1234.56",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            ),
        ]

        result = ocr_engine.call_llm(
            make_image_base64(),
            "提取发票字段",
            expected_fields=["发票号码", "开票日期", "价税合计金额"],
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2024-01-02")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    @patch("ocr_engine.requests.post")
    def test_business_card_nested_layout_shell_maps_text_and_keeps_nicole_name(self, post):
        post.return_value = FakeResponse(
            payload={
                "data": {
                    "md_results": [
                        "Nicole Wang",
                        "Sales Manager",
                        "ACME Co., Ltd",
                        "M 13812345678",
                        "nicole@example.com",
                    ]
                }
            }
        )

        result = ocr_engine.call_llm(
            make_image_base64(),
            "提取名片字段",
            expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "Nicole Wang")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "nicole@example.com")
        self.assertEqual(post.call_count, 1)

    def test_prepare_file_data_uri_preserves_small_png_mime(self):
        png_base64 = make_image_base64("PNG")

        data_uri = ocr_engine._prepare_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))
        self.assertEqual(data_uri.split(",", 1)[1], png_base64)

    def test_prepare_file_data_uri_resizes_small_byte_high_resolution_image(self):
        high_res_base64 = make_image_base64("JPEG", size=(2000, 2000), quality=30)

        data_uri = ocr_engine._prepare_file_data_uri(high_res_base64)
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as img:
            self.assertLessEqual(max(img.size), 1280)


class IndexHandlerTest(unittest.TestCase):
    @patch("index.call_llm")
    def test_handler_passes_template_context_to_ocr_engine(self, call_llm):
        call_llm.return_value = {"发票号码": "12345678", "开票日期": "2024年1月2日"}

        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": make_image_base64(),
                        "template_type": "invoice",
                    }
                )
            },
            None,
        )

        self.assertEqual(response["statusCode"], 200)
        call_llm.assert_called_once()
        _, prompt = call_llm.call_args.args
        self.assertIn("增值税发票", prompt)
        self.assertEqual(call_llm.call_args.kwargs["template_type"], "invoice")
        self.assertIn("发票号码", call_llm.call_args.kwargs["expected_fields"])
        body = json.loads(response["body"])
        self.assertEqual(body["data"]["开票日期"], "2024-01-02")

    def test_handler_rejects_custom_template_without_effective_fields(self):
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": make_image_base64(),
                        "template_type": "custom",
                        "custom_fields": ", ,",
                    }
                )
            },
            None,
        )

        self.assertEqual(response["statusCode"], 400)
        body = json.loads(response["body"])
        self.assertIn("有效字段", body["error"])

    @patch("index.call_llm")
    def test_handler_rejects_oversized_image_before_ocr_decode(self, call_llm):
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": "a" * (index.MAX_IMAGE_SIZE + 1),
                        "template_type": "business_card",
                    }
                )
            },
            None,
        )

        self.assertEqual(response["statusCode"], 413)
        call_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
