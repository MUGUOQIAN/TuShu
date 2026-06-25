import base64
import json
import unittest
from unittest.mock import Mock, patch

import index
import ocr_engine
from config import MAX_IMAGE_SIZE


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/"
    "lH8p2wAAAABJRU5ErkJggg=="
)


def _response(payload):
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.text = json.dumps(payload, ensure_ascii=False)
    return response


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_before_validation(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码 12345678\n开票日期 2026年6月25日\n"
                    "购买方名称 北京采购有限公司\n销售方名称 上海销售有限公司\n"
                    "价税合计金额 ¥1,234.50\n税额 123.45"
                ]
            }
        }
        structured_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2026年6月25日",
                                "购买方名称": "北京采购有限公司",
                                "销售方名称": "上海销售有限公司",
                                "价税合计金额": "¥1,234.50",
                                "税额": "123.45",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[_response(layout_payload), _response(structured_payload)],
        ) as post:
            event = {
                "body": json.dumps(
                    {"template_type": "invoice", "image_base64": TINY_PNG_BASE64},
                    ensure_ascii=False,
                )
            }
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["发票号码"], "12345678")
        self.assertEqual(body["data"]["开票日期"], "2026-06-25")
        self.assertEqual(body["data"]["价税合计金额"], "1234.50")
        self.assertEqual(post.call_args_list[0].kwargs["json"]["file"].split(";")[0], "data:image/png")
        self.assertEqual(post.call_count, 2)

    def test_english_name_with_company_keyword_substring_is_not_dropped(self):
        name = ocr_engine._extract_name(
            ["Nicole Wang", "Sales Manager", "ACME Ltd", "nicole@example.com"]
        )

        self.assertEqual(name, "Nicole Wang")

    def test_oversized_image_is_rejected_before_model_call(self):
        image_base64 = base64.b64encode(b"x" * (MAX_IMAGE_SIZE + 1)).decode("ascii")
        event = {
            "body": json.dumps(
                {"template_type": "business_card", "image_base64": image_base64}
            )
        }

        with patch("index.call_llm") as call_llm:
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 400)
        self.assertFalse(body["success"])
        self.assertIn("图片超过大小限制", body["error"])
        call_llm.assert_not_called()

    def test_all_empty_fields_are_not_reported_as_success(self):
        event = {
            "body": json.dumps(
                {"template_type": "invoice", "image_base64": TINY_PNG_BASE64},
                ensure_ascii=False,
            )
        }

        with patch("index.call_llm", return_value={}):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(body["success"])
        self.assertIn("未识别到有效字段", body["error"])


if __name__ == "__main__":
    unittest.main()
