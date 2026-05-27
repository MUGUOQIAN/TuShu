import base64
import json
import unittest
from unittest.mock import patch

import ocr_engine


VALID_IMAGE_BASE64 = base64.b64encode(b"tiny image bytes").decode("ascii")
INVOICE_FIELDS = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
BUSINESS_CARD_FIELDS = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码：12345678\n开票日期：2024年1月2日\n价税合计金额：¥1,234.50\n税额：67.89"
                ]
            }
        }
        structured_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2024-01-02",
                                "购买方名称": "甲方公司",
                                "销售方名称": "乙方公司",
                                "价税合计金额": "1234.50",
                                "税额": "67.89",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post"
        ) as post:
            post.side_effect = [FakeResponse(layout_payload), FakeResponse(structured_payload)]

            result = ocr_engine.call_llm(
                VALID_IMAGE_BASE64,
                "invoice prompt",
                expected_fields=INVOICE_FIELDS,
            )

        self.assertEqual("12345678", result["发票号码"])
        self.assertEqual("1234.50", result["价税合计金额"])
        self.assertEqual(2, post.call_count)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])
        user_prompt = post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
        self.assertIn("发票号码：12345678", user_prompt)
        self.assertIn("期望字段：发票号码, 开票日期", user_prompt)

    def test_direct_business_json_short_circuits_without_second_call(self):
        layout_payload = {
            "data": {
                "发票号码": "87654321",
                "开票日期": "2024-03-04",
                "税额": "12.34",
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post"
        ) as post:
            post.return_value = FakeResponse(layout_payload)

            result = ocr_engine.call_llm(
                VALID_IMAGE_BASE64,
                "invoice prompt",
                expected_fields=INVOICE_FIELDS,
            )

        self.assertEqual("87654321", result["发票号码"])
        self.assertEqual(1, post.call_count)

    def test_business_card_uses_nested_layout_text_without_chat_call(self):
        layout_payload = {
            "data": {
                "md_results": (
                    "赵美娜 Shermin Zhao\n"
                    "销售经理\n"
                    "上海玖协机械有限公司\n"
                    "M 13812345678\n"
                    "shermin@example.com\n"
                    "上海市浦东新区"
                )
            }
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post"
        ) as post:
            post.return_value = FakeResponse(layout_payload)

            result = ocr_engine.call_llm(
                VALID_IMAGE_BASE64,
                "business card prompt",
                expected_fields=BUSINESS_CARD_FIELDS,
            )

        self.assertEqual("赵美娜 (Shermin Zhao)", result["姓名"])
        self.assertEqual("13812345678", result["手机"])
        self.assertEqual("shermin@example.com", result["邮箱"])
        self.assertEqual(1, post.call_count)


if __name__ == "__main__":
    unittest.main()
