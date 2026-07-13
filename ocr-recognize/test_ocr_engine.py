import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

from index import handler
from ocr_engine import call_llm, _prepare_image_data_uri
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def image_base64(image_format="JPEG", size=(32, 32)):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTests(unittest.TestCase):
    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_before_validation(self, mock_post):
        invoice = {
            "发票号码": "12345678",
            "开票日期": "2026-07-13",
            "购买方名称": "甲方公司",
            "销售方名称": "乙方公司",
            "价税合计金额": "100.00",
            "税额": "5.66",
        }
        mock_post.side_effect = [
            FakeResponse(
                {
                    "data": {
                        "md_results": [
                            "发票号码 12345678\n开票日期 2026年7月13日\n价税合计金额 ¥100.00"
                        ]
                    }
                }
            ),
            FakeResponse({"choices": [{"message": {"content": json.dumps(invoice, ensure_ascii=False)}}]}),
        ]

        result = call_llm(
            image_base64(),
            INVOICE_TEMPLATE,
            expected_fields=list(invoice.keys()),
            template_type="invoice",
        )

        self.assertEqual(result, invoice)
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("/layout_parsing", mock_post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", mock_post.call_args_list[1].args[0])

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_nested_layout_business_card_maps_text_fields(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "data": {
                    "md_results": "张三\n销售经理\n13800138000\nzhang@example.com\n上海示例科技有限公司"
                }
            }
        )

        result = call_llm(
            image_base64(),
            "",
            expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhang@example.com")
        self.assertEqual(mock_post.call_count, 1)

    @patch("index.call_llm", return_value={})
    def test_handler_rejects_all_empty_cleaned_result(self, _mock_call_llm):
        event = {
            "body": json.dumps(
                {
                    "image_base64": image_base64(),
                    "template_type": "invoice",
                }
            )
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(body["success"])

    def test_small_png_keeps_png_data_uri(self):
        data_uri = _prepare_image_data_uri(image_base64("PNG"))
        self.assertTrue(data_uri.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
