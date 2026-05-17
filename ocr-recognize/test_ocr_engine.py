import unittest
from unittest.mock import patch

import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_data_runs_structured_extraction(self):
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "发票号码: 12345\n开票日期: 2026年5月1日\n价税合计金额: ¥1,234.50"
                    ]
                }
            }
        )
        extraction_response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"发票号码":"12345","开票日期":"2026-05-01","价税合计金额":"1234.50"}'
                        }
                    }
                ]
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine, "_compress_base64_image", return_value="compressed"
        ), patch.object(
            ocr_engine.requests, "post", side_effect=[layout_response, extraction_response]
        ) as post:
            result = ocr_engine.call_llm(
                "raw-image",
                INVOICE_TEMPLATE,
                expected_fields=["发票号码", "开票日期", "价税合计金额"],
            )

        self.assertEqual(result["发票号码"], "12345")
        self.assertEqual(result["开票日期"], "2026-05-01")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])
        self.assertIn("发票号码: 12345", post.call_args_list[1].kwargs["json"]["messages"][1]["content"])

    def test_direct_expected_dict_short_circuits(self):
        layout_response = FakeResponse({"data": {"发票号码": "INV-001"}})

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine, "_compress_base64_image", return_value="compressed"
        ), patch.object(ocr_engine.requests, "post", return_value=layout_response) as post:
            result = ocr_engine.call_llm(
                "raw-image",
                INVOICE_TEMPLATE,
                expected_fields=["发票号码", "开票日期"],
            )

        self.assertEqual(result, {"发票号码": "INV-001"})
        self.assertEqual(post.call_count, 1)

    def test_business_card_uses_local_mapping_without_chat(self):
        layout_response = FakeResponse(
            {
                "data": {
                    "md_results": [
                        "张三",
                        "产品经理",
                        "上海某某有限公司",
                        "手机 13812345678",
                    ]
                }
            }
        )
        fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine, "_compress_base64_image", return_value="compressed"
        ), patch.object(ocr_engine.requests, "post", return_value=layout_response) as post:
            result = ocr_engine.call_llm("raw-image", "名片字段抽取", expected_fields=fields)

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海某某有限公司")
        self.assertEqual(result["职位"], "产品经理")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(post.call_count, 1)

    def test_extract_text_chunks_accepts_nested_md_results_string(self):
        chunks = ocr_engine._extract_text_chunks({"data": {"md_results": "发票号码: 12345"}})

        self.assertEqual(chunks, ["发票号码: 12345"])


if __name__ == "__main__":
    unittest.main()
