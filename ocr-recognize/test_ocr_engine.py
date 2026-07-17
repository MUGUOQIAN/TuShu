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


def _image_base64():
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_extract_text_chunks_reads_nested_string_md_results(self):
        chunks = ocr_engine._extract_text_chunks(
            {
                "data": {
                    "md_results": "张三\n13800138000",
                    "layout_details": [[{"content": "zhang@example.com"}]],
                }
            }
        )

        self.assertEqual(chunks, ["张三", "13800138000", "zhang@example.com"])

    def test_business_card_layout_wrapper_is_mapped_to_fields(self):
        layout_payload = {
            "data": {
                "md_results": "张三\n销售经理\n13800138000\nzhang@example.com",
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            return_value=FakeResponse(layout_payload),
        ):
            result = ocr_engine.call_llm(
                _image_base64(),
                "抽取名片",
                expected_fields=["姓名", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhang@example.com")
        self.assertNotIn("md_results", result)

    def test_invoice_layout_text_is_structured_with_chat_model(self):
        layout_payload = {
            "data": {
                "md_results": "发票号码 123456\n价税合计金额 ¥1,234.50",
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"发票号码":"123456","价税合计金额":"1234.50"}',
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[FakeResponse(layout_payload), FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                "抽取发票字段",
                expected_fields=["发票号码", "价税合计金额"],
                template_type="invoice",
            )

        self.assertEqual(
            result,
            {"发票号码": "123456", "价税合计金额": "1234.50"},
        )
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_handler_rejects_silent_all_empty_success(self):
        request = {
            "body": json.dumps(
                {
                    "image_base64": "mock-image",
                    "template_type": "invoice",
                }
            )
        }

        with patch("index.call_llm", return_value={}) as call_llm:
            response = index.handler(request, None)

        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(json.loads(response["body"])["success"])
        call_llm.assert_called_once()
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(
            kwargs["expected_fields"],
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )


if __name__ == "__main__":
    unittest.main()
