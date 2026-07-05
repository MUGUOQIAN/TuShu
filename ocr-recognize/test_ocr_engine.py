import base64
import io
import json
import unittest
from unittest import mock

from PIL import Image

import index
import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _image_base64(fmt="PNG", size=(2, 2)):
    image = Image.new("RGB", size, color="white")
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_with_expected_fields(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码: 12345678\n开票日期: 2026年7月5日\n价税合计金额: ¥100.00"
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
                                "开票日期": "2026-07-05",
                                "购买方名称": "",
                                "销售方名称": "",
                                "价税合计金额": "100.00",
                                "税额": "",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with mock.patch("ocr_engine.GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post",
            side_effect=[FakeResponse(layout_payload), FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                INVOICE_TEMPLATE,
                expected_fields=["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "100.00")
        self.assertEqual(post.call_count, 2)
        structure_request = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码: 12345678", structure_request["messages"][0]["content"])

    def test_small_png_keeps_png_data_uri(self):
        data_uri = ocr_engine._build_glm_file_data_uri(_image_base64(fmt="PNG"))

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_high_resolution_small_image_is_resized(self):
        data_uri = ocr_engine._build_glm_file_data_uri(
            _image_base64(fmt="JPEG", size=(2400, 1200))
        )
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertLessEqual(max(image.size), 1280)


class HandlerTest(unittest.TestCase):
    def test_empty_expected_fields_returns_unprocessable_entity(self):
        request = {
            "body": json.dumps(
                {
                    "image_base64": _image_base64(),
                    "template_type": "invoice",
                },
                ensure_ascii=False,
            )
        }

        with mock.patch("index.call_llm", return_value={"md_results": ["发票号码: 123"]}):
            response = index.handler(request, None)

        self.assertEqual(response["statusCode"], 422)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])


if __name__ == "__main__":
    unittest.main()
