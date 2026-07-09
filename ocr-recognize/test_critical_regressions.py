import base64
import io
import json
import plistlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import index
import ocr_engine


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def make_image_base64(image_format="JPEG", size=(32, 32)):
    img = Image.new("RGB", size, color=(245, 245, 245))
    output = io.BytesIO()
    img.save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrCriticalRegressionTests(unittest.TestCase):
    def test_invoice_layout_text_is_structured_with_prompt(self):
        fields = ["发票号码", "开票日期", "价税合计金额"]
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/layout_parsing"):
                self.assertTrue(json["file"].startswith("data:image/jpeg;base64,"))
                return FakeResponse(
                    {
                        "data": {
                            "md_results": (
                                "发票号码: FP001\n"
                                "开票日期: 2024年1月2日\n"
                                "价税合计金额: ¥1,234.56"
                            )
                        }
                    }
                )
            self.assertTrue(url.endswith("/chat/completions"))
            user_content = json["messages"][1]["content"]
            self.assertIn("发票号码", user_content)
            self.assertIn("OCR文本如下", user_content)
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"发票号码":"FP001","开票日期":"2024-01-02",'
                                    '"价税合计金额":"1234.56"}'
                                )
                            }
                        }
                    ]
                }
            )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                make_image_base64(),
                "请提取发票号码、开票日期、价税合计金额",
                expected_fields=fields,
                template_type="invoice",
            )

        self.assertEqual(
            result,
            {"发票号码": "FP001", "开票日期": "2024-01-02", "价税合计金额": "1234.56"},
        )
        self.assertEqual(len(calls), 2)

    def test_business_card_uses_nested_string_md_results(self):
        def fake_post(url, headers, json, timeout):
            return FakeResponse(
                {
                    "data": {
                        "md_results": (
                            "Nicole Brown\n"
                            "Sales Manager\n"
                            "M: 13912345678\n"
                            "nicole@example.com\n"
                            "Example Co., Ltd"
                        )
                    }
                }
            )

        with patch.object(ocr_engine, "GLM_API_KEY", "test-key"), patch.object(
            ocr_engine.requests, "post", side_effect=fake_post
        ):
            result = ocr_engine.call_llm(
                make_image_base64(),
                "请提取名片字段",
                expected_fields=["姓名", "公司", "职位", "手机", "邮箱"],
                template_type="business_card",
            )

        self.assertEqual(result["姓名"], "Nicole Brown")
        self.assertEqual(result["手机"], "13912345678")
        self.assertEqual(result["邮箱"], "nicole@example.com")

    def test_small_png_upload_preserves_png_mime_type(self):
        normalized, mime_type = ocr_engine._prepare_image_for_upload(
            make_image_base64("PNG")
        )

        self.assertEqual(mime_type, "image/png")
        self.assertGreater(len(normalized), 0)


class HandlerCriticalRegressionTests(unittest.TestCase):
    def test_handler_passes_expected_fields_and_template_type(self):
        captured = {}

        def fake_call_llm(image_base64, prompt, **kwargs):
            captured.update(kwargs)
            return {"发票号码": "FP001", "开票日期": "2024-01-02", "价税合计金额": "12.30"}

        event = {
            "body": json.dumps(
                {
                    "image_base64": make_image_base64(),
                    "template_type": "invoice",
                },
                ensure_ascii=False,
            )
        }

        with patch.object(index, "call_llm", side_effect=fake_call_llm):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(captured["template_type"], "invoice")
        self.assertIn("发票号码", captured["expected_fields"])

    def test_handler_rejects_all_empty_success_response(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": make_image_base64(),
                    "template_type": "invoice",
                },
                ensure_ascii=False,
            )
        }

        with patch.object(index, "call_llm", return_value={}):
            response = index.handler(event, None)

        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 422)
        self.assertFalse(body["success"])
        self.assertIn("data", body)


class MobileConfigRegressionTests(unittest.TestCase):
    def test_android_release_manifest_declares_network_and_camera(self):
        manifest = ROOT / "tushu" / "android" / "app" / "src" / "main" / "AndroidManifest.xml"
        root = ET.parse(manifest).getroot()
        android_ns = "{http://schemas.android.com/apk/res/android}"
        permissions = {
            node.attrib.get(f"{android_ns}name")
            for node in root.findall("uses-permission")
        }

        self.assertIn("android.permission.INTERNET", permissions)
        self.assertIn("android.permission.CAMERA", permissions)

    def test_ios_plist_declares_camera_and_photo_usage(self):
        plist_path = ROOT / "tushu" / "ios" / "Runner" / "Info.plist"
        with plist_path.open("rb") as fh:
            plist = plistlib.load(fh)

        self.assertTrue(plist["NSCameraUsageDescription"])
        self.assertTrue(plist["NSPhotoLibraryUsageDescription"])


if __name__ == "__main__":
    unittest.main()
