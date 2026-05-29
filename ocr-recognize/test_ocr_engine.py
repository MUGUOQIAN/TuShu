import base64
import unittest
from unittest.mock import patch

from ocr_engine import call_llm
from prompt_templates import INVOICE_TEMPLATE


class MockResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def test_business_card_layout_data_is_mapped_to_fields(self):
        image_base64 = base64.b64encode(b"small image").decode("utf-8")
        layout_payload = {
            "data": {
                "md_results": [
                    "上海玖协机械有限公司\n"
                    "赵美娜 Shermin Zhao\n"
                    "销售经理\n"
                    "M 13812345678\n"
                    "shermin@example.com\n"
                    "地址 上海市浦东新区"
                ],
            },
        }

        with patch("ocr_engine.GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=MockResponse(layout_payload)
        ):
            result = call_llm(
                image_base64,
                "prompt",
                expected_fields=["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "赵美娜 (Shermin Zhao)")
        self.assertEqual(result["公司"], "上海玖协机械有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "shermin@example.com")

    def test_invoice_layout_text_is_structured_with_chat_model(self):
        image_base64 = base64.b64encode(b"small image").decode("utf-8")
        layout_payload = {
            "data": {
                "md_results": [
                    "增值税发票\n"
                    "发票号码 12345678\n"
                    "开票日期 2026年05月29日\n"
                    "购买方名称 测试买方\n"
                    "销售方名称 测试卖方\n"
                    "价税合计金额 ¥100.00\n"
                    "税额 ¥5.66"
                ],
            },
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"发票号码":"12345678","开票日期":"2026-05-29",'
                            '"购买方名称":"测试买方","销售方名称":"测试卖方",'
                            '"价税合计金额":"100.00","税额":"5.66"}'
                        ),
                    },
                },
            ],
        }

        with patch("ocr_engine.GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[MockResponse(layout_payload), MockResponse(chat_payload)],
        ) as mock_post:
            result = call_llm(
                image_base64,
                INVOICE_TEMPLATE,
                expected_fields=[
                    "发票号码",
                    "开票日期",
                    "购买方名称",
                    "销售方名称",
                    "价税合计金额",
                    "税额",
                ],
                template_type="invoice",
            )

        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("/layout_parsing", mock_post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", mock_post.call_args_list[1].args[0])
        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-05-29")
        self.assertEqual(result["价税合计金额"], "100.00")


if __name__ == "__main__":
    unittest.main()
