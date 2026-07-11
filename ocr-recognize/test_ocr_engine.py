import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _tiny_png_base64():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineRegressionTests(unittest.TestCase):
    def test_invoice_layout_wrapper_uses_structuring_model(self):
        image_base64 = _tiny_png_base64()
        fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return _FakeResponse({
                    "data": {
                        "md_results": "发票号码 12345678\n开票日期 2026年7月10日\n价税合计 ¥88.00",
                        "layout_details": [],
                    }
                })
            return _FakeResponse({
                "choices": [{
                    "message": {
                        "content": json_module.dumps({
                            "发票号码": "12345678",
                            "开票日期": "2026-07-10",
                            "价税合计金额": "88.00",
                        }, ensure_ascii=False)
                    }
                }]
            })

        json_module = json
        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                "请抽取发票字段",
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-07-10")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1]["file"].startswith("data:image/png;base64,"))
        self.assertIn("OCR文本", calls[1][1]["messages"][1]["content"])

    def test_handler_rejects_all_empty_result(self):
        event = {
            "body": json.dumps({
                "image_base64": _tiny_png_base64(),
                "template_type": "invoice",
            })
        }

        with patch.object(index, "call_llm", return_value={}):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(body["success"])

    def test_nicole_is_not_rejected_by_company_substring(self):
        result = ocr_engine._map_business_card_fields(["Nicole Wang", "Sales Manager"])

        self.assertEqual(result["姓名"], "Nicole Wang")


if __name__ == "__main__":
    unittest.main()
