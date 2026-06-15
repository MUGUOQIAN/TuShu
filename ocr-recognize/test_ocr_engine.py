import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine
from prompt_templates import TEMPLATE_MAP


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def make_image_base64(fmt="PNG", size=(32, 32), quality=75):
    image = Image.new("RGB", size, color="white")
    output = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality
    image.save(output, **save_kwargs)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_prompt(self):
        image_base64 = make_image_base64("PNG")
        fields = TEMPLATE_MAP["invoice"]["fields"]
        layout_payload = {
            "data": {
                "md_results": (
                    "发票号码 12345678\n"
                    "开票日期 2026年6月15日\n"
                    "购买方名称 上海测试公司\n"
                    "价税合计金额 ¥1,234.50"
                )
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2026-06-15",
                                "购买方名称": "上海测试公司",
                                "销售方名称": "",
                                "价税合计金额": "1234.50",
                                "税额": "",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[FakeResponse(layout_payload), FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                image_base64,
                TEMPLATE_MAP["invoice"]["template"],
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual("12345678", result["发票号码"])
        self.assertEqual("2026-06-15", result["开票日期"])
        self.assertEqual(2, post.call_count)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])
        self.assertTrue(post.call_args_list[0].kwargs["json"]["file"].startswith("data:image/png;base64,"))
        self.assertIn("发票号码 12345678", post.call_args_list[1].kwargs["json"]["messages"][1]["content"])

    def test_business_card_uses_nested_md_results_string_and_preserves_nicole(self):
        image_base64 = make_image_base64("JPEG")
        fields = TEMPLATE_MAP["business_card"]["fields"]
        layout_payload = {
            "data": {
                "md_results": "Nicole Cooper\nManager\nMobile 13800138000\nnicole@example.com"
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            return_value=FakeResponse(layout_payload),
        ):
            result = ocr_engine.call_llm(
                image_base64,
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=fields,
                template_type="business_card",
            )

        self.assertEqual("Nicole Cooper", result["姓名"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("nicole@example.com", result["邮箱"])

    def test_small_high_resolution_image_is_resized_even_under_byte_threshold(self):
        image_base64 = make_image_base64("JPEG", size=(2400, 1600), quality=5)

        resized_base64, mime_type = ocr_engine._prepare_image_for_upload(image_base64)
        resized_bytes = base64.b64decode(resized_base64)

        with Image.open(io.BytesIO(resized_bytes)) as resized:
            self.assertLessEqual(max(resized.size), 1280)
        self.assertEqual("image/jpeg", mime_type)


class HandlerTest(unittest.TestCase):
    def test_rejects_oversized_base64_before_calling_model(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": "a" * (index.MAX_IMAGE_SIZE + 1),
                    "template_type": "business_card",
                }
            )
        }

        with patch("index.call_llm") as call_llm:
            response = index.handler(event, None)

        self.assertEqual(400, response["statusCode"])
        self.assertIn("图片过大", response["body"])
        call_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
