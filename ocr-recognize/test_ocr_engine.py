import base64
import io
import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import index
import ocr_engine


def _image_base64(fmt="JPEG", size=(32, 32)):
    image = Image.new("RGB", size, color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTests(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_with_chat_model(self):
        layout_payload = {
            "data": {
                "md_results": "发票号码：INV-001\n开票日期：2024年1月2日\n价税合计金额：¥1,234.50\n税额：123.45"
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "INV-001",
                                "开票日期": "2024-01-02",
                                "购买方名称": "甲公司",
                                "销售方名称": "乙公司",
                                "价税合计金额": "1234.50",
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
            side_effect=[_Response(layout_payload), _Response(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                _image_base64(),
                "提取发票字段",
                expected_fields=["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "INV-001")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_extract_text_chunks_accepts_string_md_results_under_data(self):
        chunks = ocr_engine._extract_text_chunks(
            {"data": {"md_results": "姓名：Nicole Wang\n职位：Manager"}}
        )

        self.assertEqual(chunks, ["姓名：Nicole Wang", "职位：Manager"])

    def test_small_png_keeps_png_mime(self):
        data_uri = ocr_engine._prepare_file_data_uri(_image_base64("PNG"))

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_handler_passes_template_context_and_rejects_empty_result(self):
        body = {
            "image_base64": _image_base64(),
            "template_type": "invoice",
        }

        with patch(
            "index.call_llm",
            Mock(return_value={"md_results": "发票号码：INV-001"}),
        ) as call_llm:
            response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(response["statusCode"], 422)
        call_kwargs = call_llm.call_args.kwargs
        self.assertEqual(call_kwargs["template_type"], "invoice")
        self.assertIn("发票号码", call_kwargs["expected_fields"])


if __name__ == "__main__":
    unittest.main()
