import base64
import io
import json
import unittest
from unittest import mock

from PIL import Image

import index
import ocr_engine
from ocr_engine import InvalidImageError, call_llm


class FakeResponse:
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def make_image_base64(fmt="JPEG", size=(8, 8)):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=fmt)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTests(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_prompt(self):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                return FakeResponse({
                    "data": {
                        "md_results": "发票号码: 12345\n税额: ¥12.30",
                    }
                })
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"发票号码": "12345", "税额": "12.30"}',
                        }
                    }
                ]
            })

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), \
                mock.patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            result = call_llm(
                make_image_base64(),
                "请提取发票号码和税额",
                expected_fields=["发票号码", "税额"],
                template_type="invoice",
            )

        self.assertEqual({"发票号码": "12345", "税额": "12.30"}, result)
        self.assertEqual(2, len(calls))
        self.assertTrue(calls[1][0].endswith("/chat/completions"))
        user_prompt = calls[1][1]["messages"][1]["content"]
        self.assertIn("请提取发票号码和税额", user_prompt)
        self.assertIn("发票号码: 12345", user_prompt)

    def test_expected_fields_prevent_layout_wrapper_short_circuit(self):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append(url)
            if url.endswith("/layout_parsing"):
                return FakeResponse({"data": {"md_results": "发票号码: 9988"}})
            return FakeResponse({
                "choices": [
                    {"message": {"content": '{"发票号码": "9988"}'}}
                ]
            })

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), \
                mock.patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            result = call_llm(
                make_image_base64(),
                "请提取发票号码",
                expected_fields=["发票号码"],
                template_type="invoice",
            )

        self.assertEqual({"发票号码": "9988"}, result)
        self.assertEqual(2, len(calls))

    def test_md_results_string_is_used_for_business_card(self):
        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), \
                mock.patch.object(
                    ocr_engine.requests,
                    "post",
                    return_value=FakeResponse({
                        "data": {
                            "md_results": "张三\n销售经理\n13812345678\nzhangsan@example.com",
                        }
                    }),
                ):
            result = call_llm(
                make_image_base64(),
                "请提取名片",
                expected_fields=["姓名", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual("张三", result["姓名"])
        self.assertEqual("13812345678", result["手机"])
        self.assertEqual("zhangsan@example.com", result["邮箱"])

    def test_small_png_keeps_png_data_uri(self):
        captured_payloads = []

        def fake_post(url, headers, json, timeout):
            captured_payloads.append(json)
            return FakeResponse({"md_results": "张三\n13812345678"})

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), \
                mock.patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            call_llm(
                make_image_base64(fmt="PNG"),
                "请提取名片",
                expected_fields=["姓名", "手机"],
                template_type="business_card",
            )

        self.assertTrue(captured_payloads[0]["file"].startswith("data:image/png;base64,"))

    def test_small_but_high_resolution_image_is_resized(self):
        captured_payloads = []

        def fake_post(url, headers, json, timeout):
            captured_payloads.append(json)
            return FakeResponse({"md_results": "张三\n13812345678"})

        with mock.patch.object(ocr_engine, "GLM_API_KEY", "test-key"), \
                mock.patch.object(ocr_engine.requests, "post", side_effect=fake_post):
            call_llm(
                make_image_base64(size=(2000, 100)),
                "请提取名片",
                expected_fields=["姓名", "手机"],
                template_type="business_card",
            )

        uploaded = captured_payloads[0]["file"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(uploaded))) as img:
            self.assertLessEqual(max(img.size), 1280)


class HandlerTests(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type(self):
        event = {
            "body": json.dumps({
                "image_base64": "image-data",
                "template_type": "invoice",
            })
        }
        with mock.patch.object(index, "call_llm", return_value={"发票号码": "12345"}) as mock_call:
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(200, response["statusCode"])
        self.assertTrue(body["success"])
        self.assertEqual("12345", body["data"]["发票号码"])
        kwargs = mock_call.call_args.kwargs
        self.assertEqual(["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"], kwargs["expected_fields"])
        self.assertEqual("invoice", kwargs["template_type"])

    def test_handler_rejects_all_empty_results(self):
        event = {
            "body": json.dumps({
                "image_base64": "image-data",
                "template_type": "business_card",
            })
        }
        with mock.patch.object(index, "call_llm", return_value={}):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(422, response["statusCode"])
        self.assertFalse(body["success"])

    def test_handler_reports_invalid_images_as_bad_request(self):
        event = {
            "body": json.dumps({
                "image_base64": "not-image",
                "template_type": "business_card",
            })
        }
        with mock.patch.object(index, "call_llm", side_effect=InvalidImageError("图片数据不是有效的Base64")):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(400, response["statusCode"])
        self.assertFalse(body["success"])

    def test_custom_template_requires_real_fields(self):
        event = {
            "body": json.dumps({
                "image_base64": "image-data",
                "template_type": "custom",
                "custom_fields": ", ,",
            })
        }
        response = index.handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(400, response["statusCode"])
        self.assertFalse(body["success"])


if __name__ == "__main__":
    unittest.main()
