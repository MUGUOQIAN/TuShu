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


def _image_base64(fmt="JPEG"):
    image = Image.new("RGB", (32, 32), "white")
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self._old_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_key

    def test_invoice_layout_text_is_structured_with_prompt(self):
        posts = []

        def fake_post(url, headers=None, json=None, timeout=None):
            posts.append({"url": url, "json": json})
            if url.endswith("/layout_parsing"):
                return FakeResponse({
                    "data": {
                        "md_results": "发票号码: 12345678\n开票日期: 2026年6月4日\n价税合计金额: ¥1,234.56"
                    }
                })
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": json_dumps({
                                "发票号码": "12345678",
                                "开票日期": "2026-06-04",
                                "购买方名称": "甲方公司",
                                "销售方名称": "乙方公司",
                                "价税合计金额": "1234.56",
                                "税额": "123.45",
                            })
                        }
                    }
                ]
            })

        with patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            result = ocr_engine.call_llm(
                _image_base64(),
                TEMPLATE_MAP["invoice"]["template"],
                expected_fields=TEMPLATE_MAP["invoice"]["fields"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.56")
        self.assertEqual(len(posts), 2)
        self.assertTrue(posts[0]["json"]["file"].startswith("data:image/jpeg;base64,"))
        self.assertIn("发票号码: 12345678", posts[1]["json"]["messages"][1]["content"])

    def test_business_card_reads_string_md_results(self):
        with patch.object(
            ocr_engine.requests,
            "post",
            return_value=FakeResponse({
                "md_results": "张三\n上海示例科技有限公司\n销售经理\n手机: 13812345678\nzhangsan@example.com"
            }),
        ):
            result = ocr_engine.call_llm(
                _image_base64(),
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=TEMPLATE_MAP["business_card"]["fields"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")

    def test_small_png_uses_png_mime_in_data_uri(self):
        captured_payloads = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured_payloads.append(json)
            return FakeResponse({"md_results": "李四\n产品经理\n13912345678"})

        with patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            ocr_engine.call_llm(
                _image_base64("PNG"),
                TEMPLATE_MAP["business_card"]["template"],
                expected_fields=TEMPLATE_MAP["business_card"]["fields"],
                template_type="business_card",
            )

        self.assertTrue(captured_payloads[0]["file"].startswith("data:image/png;base64,"))

    def test_handler_passes_expected_fields_and_template_type(self):
        def fake_call_llm(image_base64, prompt, model=ocr_engine.PRIMARY_MODEL, expected_fields=None, template_type="business_card"):
            self.assertEqual(expected_fields, TEMPLATE_MAP["invoice"]["fields"])
            self.assertEqual(template_type, "invoice")
            return {
                "发票号码": "12345678",
                "开票日期": "2026年6月4日",
                "购买方名称": "甲方公司",
                "销售方名称": "乙方公司",
                "价税合计金额": "¥1,234.56",
                "税额": "123.45",
            }

        event = {
            "body": json.dumps({
                "image_base64": _image_base64(),
                "template_type": "invoice",
            })
        }
        with patch.object(index, "call_llm", side_effect=fake_call_llm):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["开票日期"], "2026-06-04")
        self.assertEqual(body["data"]["价税合计金额"], "1234.56")

    def test_custom_template_requires_non_empty_fields(self):
        event = {
            "body": json.dumps({
                "image_base64": _image_base64(),
                "template_type": "custom",
                "custom_fields": " , ",
            })
        }

        response = index.handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 400)
        self.assertFalse(body["success"])


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
