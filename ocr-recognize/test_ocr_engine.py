import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


INVOICE_FIELDS = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def image_base64(fmt="PNG", size=(16, 16)):
    image = Image.new("RGB", size, color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTests(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "发票号码：12345678\n开票日期：2026年7月6日\n价税合计金额：¥123.45"
                    ],
                    "layout_details": [],
                }
            }
        )
        structure_response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "发票号码": "12345678",
                                    "开票日期": "2026-07-06",
                                    "购买方名称": "测试购买方",
                                    "销售方名称": "测试销售方",
                                    "价税合计金额": "123.45",
                                    "税额": "11.22",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

        with (
            patch.object(ocr_engine, "GLM_API_KEY", "test-key"),
            patch.object(
                ocr_engine.requests,
                "post",
                side_effect=[layout_response, structure_response],
            ) as post,
        ):
            result = ocr_engine.call_llm(
                image_base64(),
                INVOICE_TEMPLATE,
                expected_fields=INVOICE_FIELDS,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "123.45")
        self.assertEqual(post.call_count, 2)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])
        structure_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码：12345678", structure_payload["messages"][1]["content"])

    def test_direct_business_fields_short_circuit_without_second_call(self):
        layout_response = FakeResponse(
            {
                "data": {
                    "发票号码": "INV-001",
                    "开票日期": "2026-07-06",
                }
            }
        )

        with (
            patch.object(ocr_engine, "GLM_API_KEY", "test-key"),
            patch.object(ocr_engine.requests, "post", return_value=layout_response) as post,
        ):
            result = ocr_engine.call_llm(
                image_base64(),
                INVOICE_TEMPLATE,
                expected_fields=INVOICE_FIELDS,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "INV-001")
        self.assertEqual(post.call_count, 1)

    def test_small_png_data_uri_preserves_png_mime(self):
        encoded = image_base64(fmt="PNG", size=(16, 16))

        data_uri = ocr_engine._prepare_file_data_uri(encoded)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))
        self.assertEqual(data_uri.split(",", 1)[1], encoded)

    def test_high_resolution_small_image_is_resized_to_jpeg(self):
        encoded = image_base64(fmt="PNG", size=(2000, 100))

        data_uri = ocr_engine._prepare_file_data_uri(encoded)

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        resized_bytes = base64.b64decode(data_uri.split(",", 1)[1])
        with Image.open(io.BytesIO(resized_bytes)) as resized:
            self.assertLessEqual(max(resized.size), 1280)


class IndexTests(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type(self):
        body = {
            "image_base64": image_base64(),
            "template_type": "invoice",
        }

        with patch.object(index, "call_llm", return_value={"发票号码": "INV-001"}) as call:
            response = index.handler({"body": json.dumps(body), "queryParameters": {}}, None)

        self.assertEqual(response["statusCode"], 200)
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs["expected_fields"], INVOICE_FIELDS)
        self.assertEqual(call.call_args.kwargs["template_type"], "invoice")
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["data"]["发票号码"], "INV-001")


if __name__ == "__main__":
    unittest.main()
