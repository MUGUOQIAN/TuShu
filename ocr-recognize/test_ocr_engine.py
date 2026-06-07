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


def _image_base64(image_format="JPEG", size=(16, 16)) -> str:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse(
                    {
                        "data": {
                            "md_results": [
                                "发票号码: 12345678\n"
                                "开票日期: 2026年6月7日\n"
                                "购买方名称: 甲方公司\n"
                                "销售方名称: 乙方公司\n"
                                "价税合计金额: ¥100.50\n"
                                "税额: ¥5.50"
                            ],
                            "layout_details": [],
                        }
                    }
                )
            if url.endswith("/chat/completions"):
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_module.dumps(
                                        {
                                            "发票号码": "12345678",
                                            "开票日期": "2026年6月7日",
                                            "购买方名称": "甲方公司",
                                            "销售方名称": "乙方公司",
                                            "价税合计金额": "¥100.50",
                                            "税额": "¥5.50",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        json_module = json
        body = {
            "image_base64": _image_base64(),
            "template_type": "invoice",
        }
        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            response = index.handler({"body": json.dumps(body, ensure_ascii=False)}, None)

        self.assertEqual(response["statusCode"], 200)
        payload = json.loads(response["body"])
        self.assertTrue(payload["success"])
        self.assertEqual(
            payload["data"],
            {
                "发票号码": "12345678",
                "开票日期": "2026-06-07",
                "购买方名称": "甲方公司",
                "销售方名称": "乙方公司",
                "价税合计金额": "100.50",
                "税额": "5.50",
            },
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("发票号码: 12345678", calls[1][1]["messages"][1]["content"])

    def test_business_card_layout_shell_maps_from_nested_text(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            self.assertTrue(url.endswith("/layout_parsing"))
            return FakeResponse(
                {
                    "data": {
                        "md_results": ["上海玖协机械有限公司\n赵美娜 Shermin Zhao\n销售经理\n13800138000\nshermin@example.com"],
                        "layout_details": [],
                    }
                }
            )

        body = {
            "image_base64": _image_base64(),
            "template_type": "business_card",
        }
        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            response = index.handler({"body": json.dumps(body, ensure_ascii=False)}, None)

        payload = json.loads(response["body"])
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["姓名"], "赵美娜 (Shermin Zhao)")
        self.assertEqual(payload["data"]["公司"], "上海玖协机械有限公司")
        self.assertEqual(payload["data"]["职位"], "销售经理")
        self.assertEqual(payload["data"]["手机"], "13800138000")

    def test_png_input_keeps_png_data_uri(self):
        data_uri = ocr_engine._prepare_image_data_uri(_image_base64("PNG"))
        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_small_high_resolution_jpeg_is_resized(self):
        original_b64 = _image_base64("JPEG", size=(3000, 12))
        self.assertLess(len(base64.b64decode(original_b64)), 300 * 1024)

        data_uri = ocr_engine._prepare_image_data_uri(original_b64)
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as img:
            self.assertLessEqual(max(img.size), 1280)


if __name__ == "__main__":
    unittest.main()
