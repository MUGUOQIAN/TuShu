import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine
from prompt_templates import TEMPLATE_MAP


INVOICE_FIELDS = TEMPLATE_MAP["invoice"]["fields"]
INVOICE_PROMPT = TEMPLATE_MAP["invoice"]["template"]


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _image_base64(fmt="PNG", size=(1, 1)):
    image = Image.new("RGB", size, color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.image_base64 = _image_base64()

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_layout_wrapper_is_structured_from_ocr_text(self, post):
        post.side_effect = [
            _FakeResponse(
                {
                    "data": {
                        "md_results": [
                            "发票号码: 12345678\n开票日期: 2026年7月12日\n价税合计金额: ¥88.50"
                        ]
                    }
                }
            ),
            _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "12345678",
                                        "开票日期": "2026-07-12",
                                        "购买方名称": "",
                                        "销售方名称": "",
                                        "价税合计金额": "88.50",
                                        "税额": "",
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
            self.image_base64,
            INVOICE_PROMPT,
            expected_fields=INVOICE_FIELDS,
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "88.50")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_direct_business_fields_short_circuit_without_second_call(self, post):
        post.return_value = _FakeResponse(
            {
                "data": {
                    "发票号码": "87654321",
                    "md_results": ["layout text should not be used"],
                }
            }
        )

        result = ocr_engine.call_llm(
            self.image_base64,
            INVOICE_PROMPT,
            expected_fields=INVOICE_FIELDS,
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "87654321")
        self.assertEqual(post.call_count, 1)

    def test_small_png_keeps_png_data_uri(self):
        data_uri = ocr_engine._prepare_image_data_uri(self.image_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_english_name_does_not_treat_co_substring_as_company(self):
        result = ocr_engine._map_business_card_fields(["Nicole Smith", "Manager"])

        self.assertEqual(result["姓名"], "Nicole Smith")


class HandlerRegressionTests(unittest.TestCase):
    def _decode(self, response):
        return response["statusCode"], json.loads(response["body"])

    @patch("index.call_llm", return_value={})
    def test_empty_invoice_extraction_is_not_reported_as_success(self, call_llm):
        status, body = self._decode(
            index.handler(
                {
                    "body": json.dumps(
                        {
                            "image_base64": _image_base64(),
                            "template_type": "invoice",
                        }
                    )
                },
                None,
            )
        )

        self.assertEqual(status, 422)
        self.assertFalse(body["success"])
        call_llm.assert_called_once()

    @patch("index.call_llm", return_value={"发票号码": "123"})
    def test_handler_passes_expected_fields_to_llm(self, call_llm):
        status, body = self._decode(
            index.handler(
                {
                    "body": json.dumps(
                        {
                            "image_base64": _image_base64(),
                            "template_type": "invoice",
                        }
                    )
                },
                None,
            )
        )

        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["expected_fields"], INVOICE_FIELDS)
        self.assertEqual(kwargs["template_type"], "invoice")


if __name__ == "__main__":
    unittest.main()
