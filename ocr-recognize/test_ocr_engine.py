import unittest
from unittest.mock import patch

import ocr_engine


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


class GlmOcrParsingTests(unittest.TestCase):
    def test_invoice_uses_ocr_text_then_structured_extraction(self):
        layout_response = _FakeResponse(
            {
                "md_results": "发票号码: 12345678\n价税合计金额: ¥100.00",
                "layout_details": [],
            }
        )
        chat_response = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"发票号码":"12345678","价税合计金额":"100.00"}'
                        }
                    }
                ]
            }
        )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=[layout_response, chat_response]
        ) as post:
            result = ocr_engine.call_llm(
                "aGVsbG8=",
                "请抽取发票字段",
                expected_fields=["发票号码", "价税合计金额"],
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "100.00")
        self.assertEqual(post.call_count, 2)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])
        self.assertEqual(
            post.call_args_list[1].kwargs["json"]["response_format"],
            {"type": "json_object"},
        )

    def test_layout_result_wrapper_does_not_skip_business_card_mapping(self):
        layout_response = _FakeResponse(
            {
                "result": {"raw_api_response": "not a business result"},
                "md_results": "张三\n销售经理\n13800138000\nzhangsan@example.com",
            }
        )
        fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", return_value=layout_response
        ) as post:
            result = ocr_engine.call_llm(
                "aGVsbG8=",
                "请抽取名片字段",
                expected_fields=fields,
            )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")

    def test_md_results_string_is_extracted_as_text_chunk(self):
        chunks = ocr_engine._extract_text_chunks({"md_results": "第一行\n第二行"})

        self.assertEqual(chunks, ["第一行\n第二行"])


if __name__ == "__main__":
    unittest.main()
