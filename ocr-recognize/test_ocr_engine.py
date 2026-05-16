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


class OcrEngineTest(unittest.TestCase):
    def test_invoice_uses_layout_text_for_structured_extraction(self):
        layout_payload = {
            "md_results": [
                "发票号码: 12345678\n价税合计金额: ¥45.67\n开票日期: 2026年5月16日"
            ]
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"发票号码":"12345678","价税合计金额":"45.67","开票日期":"2026-05-16"}'
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[_FakeResponse(layout_payload), _FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                "ZmFrZQ==",
                "请提取发票号码、价税合计金额、开票日期。",
                expected_fields=["发票号码", "价税合计金额", "开票日期"],
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "45.67")
        self.assertEqual(post.call_count, 2)
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(second_payload["response_format"], {"type": "json_object"})
        user_content = second_payload["messages"][0]["content"]
        self.assertIn("请提取发票号码", user_content)
        self.assertIn("发票号码: 12345678", user_content)

    def test_nested_layout_result_is_not_treated_as_final_business_json(self):
        layout_payload = {
            "data": {
                "md_results": ["客户名称: 上海测试公司\n联系人: 李四"]
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"客户名称":"上海测试公司","联系人":"李四"}'
                    }
                }
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post",
            side_effect=[_FakeResponse(layout_payload), _FakeResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                "ZmFrZQ==",
                "请提取客户名称、联系人。",
                expected_fields=["客户名称", "联系人"],
            )

        self.assertEqual(result, {"客户名称": "上海测试公司", "联系人": "李四"})
        self.assertEqual(post.call_count, 2)

    def test_business_card_keeps_single_ocr_mapping_path(self):
        layout_payload = {
            "md_results": [
                "张三\n上海测试有限公司\n销售经理\n13800138000\nzhangsan@example.com"
            ]
        }

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch(
            "ocr_engine.requests.post", return_value=_FakeResponse(layout_payload)
        ) as post:
            result = ocr_engine.call_llm(
                "ZmFrZQ==",
                "请提取名片字段。",
                expected_fields=ocr_engine.BUSINESS_CARD_FIELDS,
            )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
