import base64
import io
import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import index
import ocr_engine
from config import MAX_IMAGE_SIZE
from prompt_templates import TEMPLATE_MAP


SMALL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/6X7qN8AAAAASUVORK5CYII="
)


def mock_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = json.dumps(payload, ensure_ascii=False)
    return response


def _jpeg_base64(width=20, height=20):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.api_key_patch = patch.object(ocr_engine, "GLM_API_KEY", "test-key")
        self.api_key_patch.start()

    def tearDown(self):
        self.api_key_patch.stop()

    def test_image_pixel_limit_is_checked_before_conversion(self):
        with self.assertRaisesRegex(ValueError, "图片像素尺寸过大"):
            ocr_engine._prepare_image_data_uri(SMALL_PNG_BASE64, max_pixels=0)

    def test_small_png_keeps_png_mime_type(self):
        data_uri = ocr_engine._prepare_image_data_uri(SMALL_PNG_BASE64)
        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_high_resolution_small_jpeg_is_resized(self):
        # 低字节但高分辨率 JPEG 仍需按长边缩放，避免 OCR 超时/失败。
        large_jpeg = _jpeg_base64(2200, 1600)
        data_uri = ocr_engine._prepare_image_data_uri(large_jpeg, max_edge=1280)
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as img:
            self.assertLessEqual(max(img.size), 1280)

    def test_english_name_with_co_substring_is_kept(self):
        for name in (
            "Nicole Wang",
            "Lincoln Park",
            "Marco Rossi",
            "Addison Wang",
            "Broadway Chen",
            "Ismail Hassan",
            "Maile Chen",
            "Alex Groom",
        ):
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(
                    [
                        name,
                        "Sales Manager",
                        "13800138000",
                        "person@example.com",
                    ]
                )
                self.assertEqual(name, result["姓名"])

    def test_two_char_cn_name_with_lu_is_kept_and_not_used_as_address(self):
        for name in ("张路", "马路"):
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(
                    [
                        "上海玖协机械有限公司",
                        name,
                        "销售经理",
                        "13800138000",
                        "中山路128号",
                    ]
                )
                self.assertEqual(name, result["姓名"])
                self.assertEqual("中山路128号", result["地址"])

    def test_street_line_still_is_not_treated_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "中山路128号",
                "销售经理",
                "13800138000",
            ]
        )
        self.assertEqual("", result["姓名"])
        self.assertEqual("中山路128号", result["地址"])

    def test_address_label_still_is_not_treated_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Add: 88 West Road",
                "Sales Manager",
                "13800138000",
            ]
        )
        self.assertEqual("", result["姓名"])

    def test_email_label_still_is_not_treated_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Email: person@example.com",
                "Sales Manager",
                "13800138000",
            ]
        )
        self.assertNotIn("@", result["姓名"])
        self.assertNotEqual("person@example.com", result["姓名"])

    def test_extract_text_chunks_reads_nested_string_layout(self):
        chunks = ocr_engine._extract_text_chunks(
            {
                "data": {
                    "md_results": "发票号码 123456\n价税合计金额 100.00",
                    "layout_details": [[{"content": "购买方名称 测试公司"}]],
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

    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_expected_fields(self, post):
        fields = TEMPLATE_MAP["invoice"]["fields"]
        post.side_effect = [
            mock_response(
                {
                    "data": {
                        "md_results": (
                            "发票号码：3100261130\n"
                            "开票日期：2026年6月19日\n"
                            "购买方名称：上海测试科技有限公司\n"
                            "销售方名称：北京样例服务有限公司\n"
                            "价税合计：¥123.45\n"
                            "税额：11.23"
                        ),
                        "layout_details": [],
                    }
                }
            ),
            mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "3100261130",
                                        "开票日期": "2026-06-19",
                                        "购买方名称": "上海测试科技有限公司",
                                        "销售方名称": "北京样例服务有限公司",
                                        "价税合计金额": "123.45",
                                        "税额": "11.23",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            ),
        ]

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            TEMPLATE_MAP["invoice"]["template"],
            expected_fields=fields,
            template_type="invoice",
        )

        self.assertEqual("3100261130", result["发票号码"])
        self.assertEqual("123.45", result["价税合计金额"])
        self.assertEqual(2, post.call_count)
        layout_payload = post.call_args_list[0].kwargs["json"]
        self.assertTrue(layout_payload["file"].startswith("data:image/png;base64,"))
        structure_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("OCR文本如下", structure_payload["messages"][1]["content"])

    @patch("ocr_engine.requests.post")
    def test_business_card_uses_nested_md_results_string_without_chat_call(self, post):
        post.return_value = mock_response(
            {
                "data": {
                    "md_results": "Nicole Wang\nManager\n13800138000\nnicole@example.com",
                    "layout_details": [],
                }
            }
        )

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            TEMPLATE_MAP["business_card"]["template"],
            expected_fields=TEMPLATE_MAP["business_card"]["fields"],
            template_type="business_card",
        )

        self.assertEqual("Nicole Wang", result["姓名"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("nicole@example.com", result["邮箱"])
        self.assertEqual(1, post.call_count)

    @patch("ocr_engine.requests.post")
    def test_custom_field_cannot_collide_with_layout_metadata(self, post):
        post.side_effect = [
            mock_response(
                {
                    "id": "layout-request-id",
                    "model": "glm-ocr",
                    "md_results": "产品模型 GLM-5",
                }
            ),
            mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"model":"GLM-5"}',
                            }
                        }
                    ]
                }
            ),
        ]

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            "抽取 model",
            expected_fields=["model"],
            template_type="custom",
        )

        self.assertEqual(result, {"model": "GLM-5"})
        self.assertEqual(post.call_count, 2)


class HandlerTest(unittest.TestCase):
    def test_handler_rejects_oversized_base64_before_calling_llm(self):
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": "A" * (MAX_IMAGE_SIZE + 1),
                        "template_type": "business_card",
                    }
                )
            },
            None,
        )

        self.assertEqual(413, response["statusCode"])
        self.assertFalse(json.loads(response["body"])["success"])

    @patch("index.call_llm")
    def test_handler_passes_template_context_and_rejects_empty_result(self, call_llm):
        call_llm.return_value = {}
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": SMALL_PNG_BASE64,
                        "template_type": "invoice",
                    }
                )
            },
            None,
        )

        body = json.loads(response["body"])
        self.assertEqual(422, response["statusCode"])
        self.assertFalse(body["success"])
        self.assertIn("未识别到有效字段", body["error"])
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(
            kwargs["expected_fields"],
            TEMPLATE_MAP["invoice"]["fields"],
        )


if __name__ == "__main__":
    unittest.main()
