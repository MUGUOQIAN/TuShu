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


def make_image_base64(image_format="JPEG", size=(20, 20), quality=90):
    image = Image.new("RGB", size, color=(245, 245, 245))
    output = io.BytesIO()
    save_kwargs = {"format": image_format}
    if image_format.upper() in {"JPEG", "JPG"}:
        save_kwargs["quality"] = quality
    image.save(output, **save_kwargs)
    return base64.b64encode(output.getvalue()).decode("utf-8"), output.getvalue()


def open_base64_image(image_base64):
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))


class OcrEngineRegressionTest(unittest.TestCase):
    def test_small_png_is_normalized_to_jpeg_for_data_uri(self):
        image_base64, _ = make_image_base64("PNG")

        result = ocr_engine._compress_base64_image(image_base64)

        with open_base64_image(result) as image:
            self.assertEqual("JPEG", image.format)

    def test_small_high_resolution_jpeg_is_still_resized(self):
        image_base64, image_bytes = make_image_base64("JPEG", size=(3000, 1000), quality=45)
        self.assertLessEqual(len(image_bytes), 300 * 1024)

        result = ocr_engine._compress_base64_image(image_base64)

        with open_base64_image(result) as image:
            self.assertLessEqual(max(image.size), 1280)

    def test_english_names_with_company_substrings_are_not_dropped(self):
        result = ocr_engine._map_business_card_fields(
            ["Nicole Wang", "Sales Manager", "nicole.wang@example.com"]
        )

        self.assertEqual("Nicole Wang", result["姓名"])

    def test_company_suffix_is_not_misread_as_name(self):
        name = ocr_engine._extract_name(["Shanghai Jiuxie Machinery Co Ltd"])

        self.assertEqual("", name)

    def test_extract_text_chunks_reads_nested_layout_wrappers(self):
        chunks = ocr_engine._extract_text_chunks(
            {
                "data": {
                    "md_results": "发票号码: 00012345\n开票日期: 2024年1月2日",
                    "layout_details": [[{"content": "价税合计金额: ¥1,234.56"}]],
                }
            }
        )

        self.assertEqual(
            ["发票号码: 00012345", "开票日期: 2024年1月2日", "价税合计金额: ¥1,234.56"],
            chunks,
        )

    def test_business_card_nested_layout_response_is_mapped(self):
        image_base64, _ = make_image_base64("JPEG")
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "Nicole Wang\nSales Manager\nM 13800138000\nnicole.wang@example.com"
                    ]
                }
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=layout_response
        ) as post:
            result = ocr_engine.call_llm(
                image_base64,
                "prompt",
                expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertEqual("Nicole Wang", result["姓名"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual(1, post.call_count)

    def test_invoice_layout_text_is_structured_with_chat_model(self):
        image_base64, _ = make_image_base64("JPEG")
        layout_response = FakeResponse(
            {"data": {"md_results": ["发票号码: 00012345\n开票日期: 2024年1月2日"]}}
        )
        chat_response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"发票号码": "00012345", "开票日期": "2024-01-02"},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", side_effect=[layout_response, chat_response]
        ) as post:
            result = ocr_engine.call_llm(
                image_base64,
                "请提取发票号码和开票日期",
                expected_fields=["发票号码", "开票日期"],
                template_type="invoice",
            )

        self.assertEqual({"发票号码": "00012345", "开票日期": "2024-01-02"}, result)
        self.assertEqual(2, post.call_count)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])
        chat_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码: 00012345", chat_payload["messages"][1]["content"])


class IndexRegressionTest(unittest.TestCase):
    def test_handler_passes_template_context_to_llm(self):
        event = {"body": json.dumps({"image_base64": "abc", "template_type": "business_card"})}

        with patch("index.call_llm", return_value={"姓名": "Nicole Wang"}) as call_llm:
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertTrue(body["success"])
        self.assertEqual("Nicole Wang", body["data"]["姓名"])
        self.assertEqual("business_card", call_llm.call_args.kwargs["template_type"])
        self.assertIn("姓名", call_llm.call_args.kwargs["expected_fields"])


if __name__ == "__main__":
    unittest.main()
