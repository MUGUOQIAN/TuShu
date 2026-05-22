import base64
import unittest
from unittest.mock import patch

import ocr_engine


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def tiny_image_base64():
    return base64.b64encode(b"tiny").decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def test_invoice_layout_shell_is_structured_from_ocr_text(self):
        fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse({
                    "data": {
                        "md_results": [
                            "发票号码 12345678\n"
                            "开票日期 2026年5月1日\n"
                            "购买方名称 上海测试有限公司\n"
                            "销售方名称 北京供应商有限公司\n"
                            "价税合计 ¥1,234.50\n"
                            "税额 34.50"
                        ],
                        "layout_details": [],
                    }
                })

            self.assertTrue(url.endswith("/chat/completions"))
            user_content = json["messages"][1]["content"]
            self.assertIn("发票号码", user_content)
            self.assertIn("12345678", user_content)
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"发票号码":"12345678","开票日期":"2026-05-01",'
                                '"购买方名称":"上海测试有限公司","销售方名称":"北京供应商有限公司",'
                                '"价税合计金额":"1234.50","税额":"34.50"}'
                            )
                        }
                    }
                ]
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"):
            with patch.object(ocr_engine.requests, "post", side_effect=fake_post):
                result = ocr_engine.call_llm(
                    tiny_image_base64(),
                    "提取发票字段",
                    expected_fields=fields,
                )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "1234.50")
        self.assertEqual(len(calls), 2)

    def test_business_card_layout_shell_uses_local_mapping(self):
        fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            return FakeResponse({
                "data": {
                    "md_results": [
                        "上海玖协机械有限公司",
                        "赵美娜 Shermin Zhao",
                        "销售经理",
                        "M 13812345678",
                        "shermin@example.com",
                        "上海市浦东新区测试路88号",
                    ],
                    "layout_details": [],
                }
            })

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"):
            with patch.object(ocr_engine.requests, "post", side_effect=fake_post):
                result = ocr_engine.call_llm(
                    tiny_image_base64(),
                    "提取名片字段",
                    expected_fields=fields,
                )

        self.assertEqual(result["姓名"], "赵美娜 (Shermin Zhao)")
        self.assertEqual(result["公司"], "上海玖协机械有限公司")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
