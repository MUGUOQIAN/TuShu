import base64
import json
import unittest
from unittest import mock

import index
import ocr_engine


class MockResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_base64 = base64.b64encode(b"fake image bytes").decode("utf-8")

    def test_invoice_layout_text_uses_structured_extraction(self):
        expected_fields = ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"]
        layout_payload = {
            "md_results": [
                "发票号码: 12345678\n开票日期: 2024年1月2日\n购买方名称: 测试采购公司\n价税合计金额: ¥100.00"
            ]
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "发票号码": "12345678",
                                "开票日期": "2024-01-02",
                                "购买方名称": "测试采购公司",
                                "销售方名称": "",
                                "价税合计金额": "100.00",
                                "税额": "",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post",
            side_effect=[MockResponse(layout_payload), MockResponse(chat_payload)],
        ) as post:
            result = ocr_engine.call_llm(
                self.image_base64,
                "提取发票字段",
                expected_fields=expected_fields,
            )

        self.assertEqual(result["发票号码"], "12345678")
        self.assertEqual(result["价税合计金额"], "100.00")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/layout_parsing", post.call_args_list[0].args[0])
        self.assertIn("/chat/completions", post.call_args_list[1].args[0])

    def test_nested_layout_data_without_expected_fields_uses_text_extraction(self):
        expected_fields = ["发票号码", "开票日期"]
        layout_payload = {
            "data": {
                "md_results": ["发票号码: 87654321\n开票日期: 2024/03/04"],
            }
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"发票号码":"87654321","开票日期":"2024-03-04"}',
                    }
                }
            ]
        }

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post",
            side_effect=[MockResponse(layout_payload), MockResponse(chat_payload)],
        ):
            result = ocr_engine.call_llm(
                self.image_base64,
                "提取发票字段",
                expected_fields=expected_fields,
            )

        self.assertEqual(result, {"发票号码": "87654321", "开票日期": "2024-03-04"})

    def test_business_card_layout_text_keeps_local_mapping(self):
        expected_fields = ["姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"]
        layout_payload = {
            "md_results": [
                "张三",
                "销售经理",
                "上海测试有限公司",
                "手机 13800138000",
                "邮箱 zhangsan@example.com",
            ]
        }

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post",
            return_value=MockResponse(layout_payload),
        ) as post:
            result = ocr_engine.call_llm(
                self.image_base64,
                "提取名片字段",
                expected_fields=expected_fields,
            )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")

    def test_direct_structured_result_short_circuits(self):
        expected_fields = ["发票号码", "开票日期"]
        layout_payload = {
            "data": {
                "发票号码": "11223344",
                "开票日期": "2024-05-06",
            }
        }

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), mock.patch(
            "ocr_engine.requests.post",
            return_value=MockResponse(layout_payload),
        ) as post:
            result = ocr_engine.call_llm(
                self.image_base64,
                "提取发票字段",
                expected_fields=expected_fields,
            )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(result, {"发票号码": "11223344", "开票日期": "2024-05-06"})


class HandlerTest(unittest.TestCase):
    def test_handler_passes_expected_fields_to_llm(self):
        body = {
            "image_base64": base64.b64encode(b"fake image bytes").decode("utf-8"),
            "template_type": "invoice",
        }

        with mock.patch(
            "index.call_llm",
            return_value={
                "发票号码": "12345678",
                "开票日期": "2024-01-02",
                "购买方名称": "",
                "销售方名称": "",
                "价税合计金额": "100.00",
                "税额": "",
            },
        ) as call:
            response = index.handler({"body": json.dumps(body, ensure_ascii=False)}, None)

        self.assertEqual(response["statusCode"], 200)
        _, kwargs = call.call_args
        self.assertEqual(
            kwargs["expected_fields"],
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )


if __name__ == "__main__":
    unittest.main()
