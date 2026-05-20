import base64
import json
import unittest
from unittest.mock import patch

import ocr_engine
from prompt_templates import INVOICE_TEMPLATE


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_prompt(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "json": json})
            if url.endswith("/layout_parsing"):
                return FakeResponse(
                    {
                        "md_results": [
                            "发票号码 12345678\n"
                            "开票日期 2026年5月20日\n"
                            "购买方名称 北京图书有限公司\n"
                            "销售方名称 上海纸张有限公司\n"
                            "价税合计金额 ¥1,234.56\n"
                            "税额 123.45"
                        ]
                    }
                )
            if url.endswith("/chat/completions"):
                user_content = json["messages"][1]["content"]
                self.assertIn("发票号码", user_content)
                self.assertIn("12345678", user_content)
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_module_dumps(
                                        {
                                            "发票号码": "12345678",
                                            "开票日期": "2026-05-20",
                                            "购买方名称": "北京图书有限公司",
                                            "销售方名称": "上海纸张有限公司",
                                            "价税合计金额": "1234.56",
                                            "税额": "123.45",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            self.fail(f"unexpected url: {url}")

        fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        image_base64 = base64.b64encode(b"fake image").decode("ascii")

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                image_base64,
                INVOICE_TEMPLATE,
                expected_fields=fields,
            )

        self.assertEqual("12345678", result["发票号码"])
        self.assertEqual("北京图书有限公司", result["购买方名称"])
        self.assertEqual(2, len(calls))
        self.assertTrue(calls[0]["url"].endswith("/layout_parsing"))
        self.assertTrue(calls[1]["url"].endswith("/chat/completions"))

    def test_business_card_still_uses_local_mapping(self):
        chunks = [
            "张三",
            "销售经理",
            "上海图书有限公司",
            "手机 13800138000",
            "zhangsan@example.com",
        ]

        result = ocr_engine._map_business_card_fields(chunks)

        self.assertEqual("张三", result["姓名"])
        self.assertEqual("销售经理", result["职位"])
        self.assertEqual("13800138000", result["手机"])


def json_module_dumps(value):
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
