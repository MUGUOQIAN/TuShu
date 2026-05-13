import base64
import json
import unittest
from unittest.mock import patch

import index
import ocr_engine


SMALL_IMAGE_BASE64 = base64.b64encode(b"small-image-placeholder").decode("utf-8")


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_invoice_uses_prompted_vision_model(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"发票号码": "INV-001", "税额": "12.30"}'
                        }
                    }
                ]
            }
        )

        result = ocr_engine.call_llm(
            SMALL_IMAGE_BASE64,
            "请提取发票号码和税额，仅返回JSON",
            template_type="invoice",
        )

        self.assertEqual(result["发票号码"], "INV-001")
        self.assertEqual(result["税额"], "12.30")
        called_url = mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_url, ocr_engine.CHAT_COMPLETIONS_URL)
        self.assertEqual(payload["model"], ocr_engine.VISION_MODEL)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertIn("请提取发票号码和税额", content[1]["text"])

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_business_card_reads_top_level_md_results_string(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "md_results": "\n".join(
                    [
                        "张三",
                        "上海测试有限公司",
                        "销售经理",
                        "手机 13800138000",
                        "zhangsan@example.com",
                        "上海市浦东新区测试路100号",
                    ]
                )
            }
        )

        result = ocr_engine.call_llm(
            SMALL_IMAGE_BASE64,
            "ignored for glm-ocr layout parsing",
            template_type="business_card",
        )

        self.assertEqual(mock_post.call_args.args[0], ocr_engine.LAYOUT_PARSING_URL)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海测试有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")

    @patch("ocr_engine.GLM_API_KEY", "test-key")
    @patch("ocr_engine.requests.post")
    def test_business_card_reads_nested_layout_payload(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "data": {
                    "md_results": "李四\n北京样例集团\n工程师\n13900139000",
                    "layout_details": [],
                }
            }
        )

        result = ocr_engine.call_llm(
            SMALL_IMAGE_BASE64,
            "ignored for glm-ocr layout parsing",
            template_type="business_card",
        )

        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["公司"], "北京样例集团")
        self.assertEqual(result["职位"], "工程师")
        self.assertEqual(result["手机"], "13900139000")
        self.assertNotIn("md_results", result)

    @patch("index.call_llm")
    def test_handler_passes_template_type_to_model_layer(self, mock_call_llm):
        mock_call_llm.return_value = {"发票号码": "INV-002"}
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": SMALL_IMAGE_BASE64,
                        "template_type": "invoice",
                    }
                )
            },
            None,
        )

        self.assertEqual(response["statusCode"], 200)
        mock_call_llm.assert_called_once()
        self.assertEqual(mock_call_llm.call_args.kwargs["template_type"], "invoice")
        body = json.loads(response["body"])
        self.assertEqual(body["data"]["发票号码"], "INV-002")


if __name__ == "__main__":
    unittest.main()
