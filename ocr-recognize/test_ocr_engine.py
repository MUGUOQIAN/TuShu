import base64
import json
import unittest
from unittest.mock import patch

import ocr_engine
from prompt_templates import TEMPLATE_MAP


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_base64 = base64.b64encode(b"small image payload").decode("utf-8")

    def test_business_card_maps_nested_layout_data(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "张三\n销售经理\n上海示例科技有限公司\n手机 13812345678\nzhangsan@example.com\n上海市浦东新区示例路1号"
                ]
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=FakeResponse(layout_payload)
        ):
            result = ocr_engine.call_llm(
                self.image_base64,
                TEMPLATE_MAP["business_card"]["template"],
                fields=TEMPLATE_MAP["business_card"]["fields"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")

    def test_invoice_uses_ocr_text_and_prompt_for_field_extraction(self):
        layout_payload = {
            "md_results": [
                "增值税专用发票\n发票号码: 12345678\n开票日期: 2026年05月14日\n"
                "购买方名称: 上海采购有限公司\n销售方名称: 北京销售有限公司\n"
                "价税合计金额: ¥1,234.56\n税额: 123.45"
            ]
        }
        extraction_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2026-05-14",
                                "购买方名称": "上海采购有限公司",
                                "销售方名称": "北京销售有限公司",
                                "价税合计金额": "1234.56",
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
            side_effect=[FakeResponse(layout_payload), FakeResponse(extraction_payload)],
        ) as post_mock:
            result = ocr_engine.call_llm(
                self.image_base64,
                TEMPLATE_MAP["invoice"]["template"],
                fields=TEMPLATE_MAP["invoice"]["fields"],
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["购买方名称"], "上海采购有限公司")
        self.assertEqual(post_mock.call_count, 2)
        extraction_request = post_mock.call_args_list[1].kwargs["json"]
        self.assertEqual(extraction_request["model"], ocr_engine.TEXT_MODEL)
        self.assertIn("OCR文本", extraction_request["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
