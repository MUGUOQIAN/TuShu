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
    def test_extract_text_chunks_reads_nested_string_layout(self):
        chunks = ocr_engine._extract_text_chunks(
            {
                "data": {
                    "md_results": "发票号码 123456\n价税合计金额 100.00",
                    "layout_details": [
                        [{"content": "购买方名称 测试公司"}]
                    ],
                }
            }
        )

        self.assertEqual(
            chunks,
            [
                "发票号码 123456",
                "价税合计金额 100.00",
                "购买方名称 测试公司",
            ],
        )

    def test_business_card_layout_shell_is_mapped(self):
        layout_payload = {
            "data": {
                "md_results": "张三\n销售经理\n13800138000",
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            return_value=FakeResponse(layout_payload),
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                "抽取名片字段",
                expected_fields=["姓名", "职位", "手机"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(post.call_count, 1)

    def test_invoice_layout_shell_uses_structuring_model(self):
        fields = ["发票号码", "价税合计金额"]
        layout_payload = {
            "data": {
                "md_results": "发票号码 123456\n价税合计金额 ¥1,234.50",
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"发票号码":"123456",'
                            '"价税合计金额":"1234.50"}'
                        ),
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[
                FakeResponse(layout_payload),
                FakeResponse(chat_payload),
            ],
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                "抽取发票字段",
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(
            result,
            {"发票号码": "123456", "价税合计金额": "1234.50"},
        )
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_custom_field_cannot_collide_with_layout_metadata(self):
        layout_payload = {
            "id": "layout-request-id",
            "model": "glm-ocr",
            "md_results": "产品模型 GLM-5",
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"model":"GLM-5"}',
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[
                FakeResponse(layout_payload),
                FakeResponse(chat_payload),
            ],
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                "抽取 model",
                expected_fields=["model"],
                template_type="custom",
            )

        self.assertEqual(result, {"model": "GLM-5"})
        self.assertEqual(post.call_count, 2)

    def test_handler_passes_template_context_and_rejects_empty_result(self):
        body = {
            "image_base64": "abc",
            "template_type": "invoice",
        }
        with patch("index.call_llm", return_value={}) as call_llm:
            response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(response["statusCode"], 422)
        payload = json.loads(response["body"])
        self.assertFalse(payload["success"])
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(
            kwargs["expected_fields"],
            [
                "发票号码",
                "开票日期",
                "购买方名称",
                "销售方名称",
                "价税合计金额",
                "税额",
            ],
        )


if __name__ == "__main__":
    unittest.main()
