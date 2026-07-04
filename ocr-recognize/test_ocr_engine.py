import base64
import io
import json
import unittest
from unittest.mock import MagicMock, patch

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


def image_base64(fmt="PNG", size=(40, 30), color=(255, 255, 255)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTests(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        fields = TEMPLATE_MAP["invoice"]["fields"]
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码 12345678\n开票日期 2026年7月4日\n价税合计金额 ¥1,234.56"
                ],
                "layout_details": [],
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2026-07-04",
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

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[FakeResponse(layout_payload), FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                image_base64("JPEG"),
                TEMPLATE_MAP["invoice"]["template"],
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.56")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_business_card_nested_layout_text_maps_fields(self):
        payload = {
            "data": {
                "md_results": "Nicole Wang\nSales Manager\nnicole@example.com\n13800138000",
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=FakeResponse(payload)
        ):
            result = ocr_engine.call_llm(
                image_base64("JPEG"),
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=TEMPLATE_MAP["business_card"]["fields"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Nicole Wang")
        self.assertEqual(result["职位"], "Sales Manager")
        self.assertEqual(result["邮箱"], "nicole@example.com")

    def test_small_png_keeps_png_data_uri(self):
        png_base64 = image_base64("PNG")

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            return_value=FakeResponse({"姓名": "张三"}),
        ) as post:
            ocr_engine.call_llm(
                png_base64,
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=TEMPLATE_MAP["business_card"]["fields"],
                template_type="business_card",
            )

        uploaded_file = post.call_args.kwargs["json"]["file"]
        self.assertTrue(uploaded_file.startswith("data:image/png;base64,"))

    def test_high_resolution_small_image_is_resized(self):
        source = image_base64("JPEG", size=(2200, 100))
        mime_type, normalized = ocr_engine._prepare_image_for_glm(source)
        decoded = base64.b64decode(normalized)

        with Image.open(io.BytesIO(decoded)) as img:
            self.assertEqual(mime_type, "image/jpeg")
            self.assertLessEqual(max(img.size), 1280)


class HandlerTests(unittest.TestCase):
    def test_handler_passes_expected_fields_and_rejects_all_empty_result(self):
        with patch("index.call_llm", return_value={}) as call_llm:
            response = index.handler(
                {
                    "body": json.dumps(
                        {
                            "image_base64": image_base64("JPEG"),
                            "template_type": "invoice",
                        }
                    )
                },
                None,
            )

        self.assertEqual(response["statusCode"], 422)
        call_llm.assert_called_once()
        self.assertEqual(call_llm.call_args.kwargs["template_type"], "invoice")
        self.assertEqual(
            call_llm.call_args.kwargs["expected_fields"],
            TEMPLATE_MAP["invoice"]["fields"],
        )

    def test_handler_rejects_oversized_payload_before_ocr(self):
        with patch("index.call_llm") as call_llm:
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

        self.assertEqual(response["statusCode"], 400)
        call_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
