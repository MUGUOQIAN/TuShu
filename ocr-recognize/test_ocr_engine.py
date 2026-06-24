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


def _image_base64(fmt="JPEG", size=(20, 20), quality=75):
    image = Image.new("RGB", size, "white")
    output = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(output, **save_kwargs)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_prepare_file_data_uri_preserves_png_mime_for_small_images(self):
        image_base64 = _image_base64("PNG")

        data_uri = ocr_engine._prepare_file_data_uri(image_base64)

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_prepare_file_data_uri_resizes_small_byte_high_resolution_images(self):
        image_base64 = _image_base64("JPEG", size=(2400, 600), quality=10)

        data_uri = ocr_engine._prepare_file_data_uri(image_base64, max_edge=1280)
        encoded = data_uri.split(",", 1)[1]
        resized_bytes = base64.b64decode(encoded)
        with Image.open(io.BytesIO(resized_bytes)) as image:
            self.assertLessEqual(max(image.size), 1280)

    def test_extract_text_chunks_reads_string_md_results_and_nested_layout(self):
        chunks = ocr_engine._extract_text_chunks({
            "data": {
                "md_results": "发票号码 123456\n价税合计金额 100.00",
                "layout_details": [[{"content": "购买方名称 测试公司"}]],
            }
        })

        self.assertIn("发票号码 123456\n价税合计金额 100.00", chunks)
        self.assertIn("购买方名称 测试公司", chunks)

    def test_invoice_layout_shell_is_structured_with_chat_model(self):
        fields = ["发票号码", "价税合计金额"]
        image_base64 = _image_base64()
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
                image_base64,
                "抽取发票字段",
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "123456")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_handler_passes_fields_and_rejects_all_empty_results(self):
        body = {
            "image_base64": "abc",
            "template_type": "invoice",
        }
        with patch("index.call_llm", return_value={}) as call_llm:
            response = index.handler({"body": json.dumps(body)}, None)

        self.assertEqual(response["statusCode"], 422)
        payload = json.loads(response["body"])
        self.assertFalse(payload["success"])
        call_llm.assert_called_once()
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(kwargs["expected_fields"], ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"])

    def test_extract_name_does_not_reject_names_containing_co(self):
        chunks = ["Sales Manager", "Nicole Scott", "nicole@example.com"]

        result = ocr_engine._map_business_card_fields(chunks)

        self.assertEqual(result["姓名"], "Nicole Scott")


if __name__ == "__main__":
    unittest.main()
