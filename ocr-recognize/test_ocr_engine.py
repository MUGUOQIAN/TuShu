import json
import unittest
from unittest.mock import patch

import index
import ocr_engine


BUSINESS_CARD_FIELDS = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]
INVOICE_FIELDS = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.glm_key_patch = patch.object(ocr_engine, "GLM_API_KEY", "test-key")
        self.glm_key_patch.start()

    def tearDown(self):
        self.glm_key_patch.stop()

    def test_business_fields_short_circuit_when_present(self):
        layout_payload = {
            "data": {
                "姓名": "张三",
                "公司": "测试科技有限公司",
                "手机": "13812345678",
            }
        }

        with patch.object(ocr_engine.requests, "post", return_value=FakeResponse(layout_payload)) as post:
            result = ocr_engine.call_llm(
                "aGVsbG8=",
                "prompt",
                expected_fields=BUSINESS_CARD_FIELDS,
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(post.call_count, 1)

    def test_business_card_layout_wrapper_is_mapped_from_text(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "张三\n总经理\n上海测试科技有限公司\n手机 13812345678\nzhangsan@example.com"
                ]
            }
        }

        with patch.object(ocr_engine.requests, "post", return_value=FakeResponse(layout_payload)) as post:
            result = ocr_engine.call_llm(
                "aGVsbG8=",
                "prompt",
                expected_fields=BUSINESS_CARD_FIELDS,
                template_type="business_card",
            )

        self.assertNotIn("md_results", result)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(post.call_count, 1)

    def test_invoice_layout_text_is_structured_with_chat_model(self):
        layout_payload = {
            "data": {
                "md_results": [
                    "发票号码 12345678\n开票日期 2026年6月18日\n购买方名称 北京采购有限公司\n"
                    "销售方名称 上海销售有限公司\n价税合计金额 ¥100.00\n税额 5.66"
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
                                "开票日期": "2026-06-18",
                                "购买方名称": "北京采购有限公司",
                                "销售方名称": "上海销售有限公司",
                                "价税合计金额": "100.00",
                                "税额": "5.66",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        responses = [FakeResponse(layout_payload), FakeResponse(structured_payload)]

        def fake_post(url, **kwargs):
            return responses.pop(0)

        with patch.object(ocr_engine.requests, "post", side_effect=fake_post) as post:
            result = ocr_engine.call_llm(
                "aGVsbG8=",
                "invoice prompt",
                expected_fields=INVOICE_FIELDS,
                template_type="invoice",
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "100.00")
        self.assertEqual(post.call_count, 2)
        self.assertIn("layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("chat/completions", post.call_args_list[1].args[0])

    def test_handler_passes_expected_fields_to_ocr_engine(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": "aGVsbG8=",
                    "template_type": "invoice",
                },
                ensure_ascii=False,
            )
        }

        with patch.object(index, "call_llm", return_value={"发票号码": "12345678"}) as call_llm:
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        call_llm.assert_called_once()
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["expected_fields"], INVOICE_FIELDS)
        self.assertEqual(kwargs["template_type"], "invoice")


if __name__ == "__main__":
    unittest.main()
