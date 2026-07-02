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


def make_image_base64(fmt="JPEG", size=(32, 32)):
    output = io.BytesIO()
    Image.new("RGB", size, color="white").save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTests(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_chat_model(self):
        image_base64 = make_image_base64()
        expected_fields = ["发票号码", "价税合计金额"]

        def fake_post(url, headers, json, timeout):
            if url.endswith("/layout_parsing"):
                self.assertTrue(json["file"].startswith("data:image/jpeg;base64,"))
                return FakeResponse(
                    {
                        "md_results": "发票号码: 12345678\n价税合计金额: ¥100.50",
                        "layout_details": [],
                    }
                )
            if url.endswith("/chat/completions"):
                user_prompt = json["messages"][1]["content"]
                self.assertIn("发票号码: 12345678", user_prompt)
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"发票号码":"12345678","价税合计金额":"100.50"}'
                                }
                            }
                        ]
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                "提取发票字段",
                expected_fields=expected_fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "100.50")

    def test_extract_text_chunks_handles_string_and_nested_md_results(self):
        chunks = ocr_engine._extract_text_chunks(
            {
                "data": {
                    "md_results": "张三\n产品经理",
                    "layout_details": [[{"content": "13800138000"}]],
                }
            }
        )

        self.assertEqual(chunks, ["张三", "产品经理", "13800138000"])

    def test_small_png_uses_png_data_uri_instead_of_jpeg_label(self):
        png_base64 = make_image_base64(fmt="PNG")

        data_uri = ocr_engine._prepare_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_compact_high_resolution_image_is_resized_to_jpeg(self):
        png_base64 = make_image_base64(fmt="PNG", size=(2048, 256))

        data_uri = ocr_engine._prepare_file_data_uri(png_base64)

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))

    def test_english_name_with_co_substring_is_not_rejected(self):
        result = ocr_engine._map_business_card_fields(["Nicole Wang", "Sales Manager"])

        self.assertEqual(result["姓名"], "Nicole Wang")


class HandlerTests(unittest.TestCase):
    def test_handler_passes_fields_and_template_type_to_llm(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": make_image_base64(),
                    "template_type": "invoice",
                }
            )
        }

        with patch("index.call_llm") as call_llm:
            call_llm.return_value = {
                "发票号码": "12345678",
                "开票日期": "2026-07-02",
                "购买方名称": "买方公司",
                "销售方名称": "卖方公司",
                "价税合计金额": "100.50",
                "税额": "5.50",
            }
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        call_llm.assert_called_once()
        kwargs = call_llm.call_args.kwargs
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertIn("发票号码", kwargs["expected_fields"])

    def test_handler_rejects_all_empty_results(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": make_image_base64(),
                    "template_type": "business_card",
                }
            )
        }

        with patch("index.call_llm", return_value={}):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 422)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])


if __name__ == "__main__":
    unittest.main()
