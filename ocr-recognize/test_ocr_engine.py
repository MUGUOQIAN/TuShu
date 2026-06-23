import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


class MockResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _image_base64(fmt="JPEG") -> str:
    image = Image.new("RGB", (24, 24), color="white")
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_uses_layout_text_for_structured_extraction(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return MockResponse(
                    {
                        "data": {
                            "md_results": (
                                "发票号码: FP001\n"
                                "开票日期: 2026年6月23日\n"
                                "购买方名称: 买方科技\n"
                                "销售方名称: 卖方科技\n"
                                "价税合计金额: ¥1,234.50\n"
                                "税额: 100.00"
                            )
                        }
                    }
                )
            return MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "发票号码": "FP001",
                                        "开票日期": "2026-06-23",
                                        "购买方名称": "买方科技",
                                        "销售方名称": "卖方科技",
                                        "价税合计金额": "1234.50",
                                        "税额": "100.00",
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        event = {
            "body": json.dumps(
                {
                    "image_base64": _image_base64(),
                    "template_type": "invoice",
                }
            )
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(200, response["statusCode"])
        self.assertTrue(body["success"])
        self.assertEqual("FP001", body["data"]["发票号码"])
        self.assertEqual("2026-06-23", body["data"]["开票日期"])
        self.assertEqual("1234.50", body["data"]["价税合计金额"])
        self.assertEqual(2, len(calls))
        self.assertTrue(calls[1][0].endswith("/chat/completions"))
        self.assertIn("发票号码", calls[1][1]["messages"][0]["content"])

    def test_business_card_extracts_nested_layout_text(self):
        captured_payloads = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_payloads.append(json)
            return MockResponse(
                {
                    "data": {
                        "md_results": (
                            "张三\n"
                            "高级经理\n"
                            "上海测试有限公司\n"
                            "手机 13812345678\n"
                            "邮箱 zhangsan@example.com"
                        )
                    }
                }
            )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                _image_base64(),
                "",
                expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertEqual("张三", result["姓名"])
        self.assertEqual("13812345678", result["手机"])
        self.assertEqual("zhangsan@example.com", result["邮箱"])
        self.assertEqual(1, len(captured_payloads))

    def test_handler_rejects_all_empty_result(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return MockResponse({"data": {"md_results": ""}})

        event = {
            "body": json.dumps(
                {
                    "image_base64": _image_base64(),
                    "template_type": "business_card",
                }
            )
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(422, response["statusCode"])
        self.assertFalse(body["success"])

    def test_small_png_keeps_png_data_uri_and_english_name(self):
        captured_payloads = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_payloads.append(json)
            return MockResponse(
                {
                    "md_results": (
                        "Nicole Wong\n"
                        "Manager\n"
                        "Acme Ltd\n"
                        "13900000000\n"
                        "nicole@example.com"
                    )
                }
            )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                _image_base64("PNG"),
                "",
                expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertTrue(captured_payloads[0]["file"].startswith("data:image/png;base64,"))
        self.assertEqual("Nicole Wong", result["姓名"])
        self.assertEqual("13900000000", result["手机"])


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
