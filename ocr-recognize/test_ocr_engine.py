import base64
import unittest
from unittest.mock import Mock, patch

import ocr_engine
from prompt_templates import TEMPLATE_MAP


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_base64 = base64.b64encode(b"small-test-image").decode("utf-8")
        self._old_api_key = ocr_engine.GLM_API_KEY
        ocr_engine.GLM_API_KEY = "test-key"

    def tearDown(self):
        ocr_engine.GLM_API_KEY = self._old_api_key

    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_prompt(self, mock_post):
        layout_response = Mock(
            status_code=200,
            text="ok",
            json=Mock(
                return_value={
                    "data": {
                        "md_results": "发票号码 12345678\n开票日期 2026年5月24日\n价税合计金额 ¥99.50"
                    }
                }
            ),
        )
        extract_response = Mock(
            status_code=200,
            text="ok",
            json=Mock(
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": '{"发票号码":"12345678","开票日期":"2026-05-24","价税合计金额":"99.50"}'
                            }
                        }
                    ]
                }
            ),
        )
        mock_post.side_effect = [layout_response, extract_response]

        result = ocr_engine.call_llm(
            self.image_base64,
            TEMPLATE_MAP["invoice"]["template"],
            expected_fields=TEMPLATE_MAP["invoice"]["fields"],
        )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["开票日期"], "2026-05-24")
        self.assertEqual(result["价税合计金额"], "99.50")
        self.assertEqual(mock_post.call_count, 2)
        chat_payload = mock_post.call_args_list[1].kwargs["json"]
        self.assertIn("发票号码", chat_payload["messages"][1]["content"])
        self.assertIn("12345678", chat_payload["messages"][1]["content"])

    @patch("ocr_engine.requests.post")
    def test_business_card_uses_string_md_results(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200,
            text="ok",
            json=Mock(
                return_value={
                    "md_results": "赵美娜 Shermin Zhao\n销售经理\n上海玖协机械有限公司\n手机 13812345678\nshermin@example.com"
                }
            ),
        )

        result = ocr_engine.call_llm(
            self.image_base64,
            TEMPLATE_MAP["business_card"]["template"],
            expected_fields=TEMPLATE_MAP["business_card"]["fields"],
        )

        self.assertEqual(result["姓名"], "赵美娜")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "shermin@example.com")
        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
