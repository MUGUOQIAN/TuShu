import base64
import io
import json
import unittest
from unittest.mock import Mock, patch

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


def image_base64(fmt="PNG", size=(2, 2)):
    image = Image.new("RGB", size, "white")
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTests(unittest.TestCase):
    def setUp(self):
        self.api_key_patch = patch.object(ocr_engine, "GLM_API_KEY", "test-key")
        self.api_key_patch.start()

    def tearDown(self):
        self.api_key_patch.stop()

    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_by_expected_fields(self, post):
        post.side_effect = [
            FakeResponse(
                {
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2026年6月5日\n价税合计金额: ¥1,234.50"
                        ]
                    }
                }
            ),
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2026-06-05",
                                        "价税合计金额": "1234.50",
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
            image_base64(),
            "请提取发票字段",
            expected_fields=["发票号码", "开票日期", "价税合计金额"],
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-06-05")
        self.assertEqual(post.call_count, 2)
        layout_payload = post.call_args_list[0].kwargs["json"]
        self.assertTrue(layout_payload["file"].startswith("data:image/png;base64,"))
        chat_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码: 12345678", chat_payload["messages"][0]["content"])

    @patch("ocr_engine.requests.post")
    def test_business_card_layout_text_uses_local_mapping_without_chat(self, post):
        post.return_value = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "张三\n产品经理\n上海示例科技有限公司\n手机 13800138000\nzhangsan@example.com"
                    ]
                }
            }
        )

        result = ocr_engine.call_llm(
            image_base64(),
            "请提取名片字段",
            expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(post.call_count, 1)

    @patch("ocr_engine.requests.post")
    def test_direct_expected_fields_are_returned_without_chat(self, post):
        post.return_value = FakeResponse(
            {
                "data": {
                    "发票号码": "87654321",
                    "开票日期": "2026-06-05",
                }
            }
        )

        result = ocr_engine.call_llm(
            image_base64("JPEG"),
            "请提取发票字段",
            expected_fields=["发票号码", "开票日期"],
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "87654321")
        self.assertEqual(post.call_count, 1)

    @patch("ocr_engine.requests.post")
    def test_small_png_keeps_png_mime_and_large_jpeg_is_resized(self, post):
        post.return_value = FakeResponse({"data": {"姓名": "张三"}})

        ocr_engine.call_llm(
            image_base64("PNG", (2, 2)),
            "请提取名片字段",
            expected_fields=["姓名"],
            template_type="business_card",
        )
        png_payload = post.call_args.kwargs["json"]
        self.assertTrue(png_payload["file"].startswith("data:image/png;base64,"))

        ocr_engine.call_llm(
            image_base64("JPEG", (2000, 1000)),
            "请提取名片字段",
            expected_fields=["姓名"],
            template_type="business_card",
        )
        jpeg_payload = post.call_args.kwargs["json"]
        self.assertTrue(jpeg_payload["file"].startswith("data:image/jpeg;base64,"))
        encoded = jpeg_payload["file"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as resized:
            self.assertLessEqual(max(resized.size), 1280)


class HandlerTests(unittest.TestCase):
    @patch("index.call_llm")
    def test_handler_passes_template_type_and_expected_fields(self, call_llm):
        call_llm.return_value = {"发票号码": "12345678", "开票日期": "2026-06-05"}

        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": image_base64(),
                        "template_type": "invoice",
                    }
                )
            },
            None,
        )

        self.assertEqual(response["statusCode"], 200)
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertIn("发票号码", kwargs["expected_fields"])


if __name__ == "__main__":
    unittest.main()
