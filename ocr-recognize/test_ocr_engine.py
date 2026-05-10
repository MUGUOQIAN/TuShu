import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GLM_API_KEY", "test-key")

import ocr_engine


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class GlmFallbackTest(unittest.TestCase):
    def test_non_dict_response_fails_instead_of_empty_success(self):
        with patch.object(ocr_engine, "_compress_base64_image", return_value="image"), patch(
            "ocr_engine.requests.post", return_value=_FakeResponse([])
        ):
            with self.assertRaisesRegex(RuntimeError, "GLM返回格式异常"):
                ocr_engine._call_glm("image", "名片")

    def test_non_business_layout_response_fails_instead_of_business_card_mapping(self):
        payload = {"md_results": ["发票号码 12345678", "价税合计金额 100.00"]}
        invoice_prompt = "字段列表：发票号码, 开票日期, 购买方名称, 销售方名称, 价税合计金额, 税额"

        with patch.object(ocr_engine, "_compress_base64_image", return_value="image"), patch(
            "ocr_engine.requests.post", return_value=_FakeResponse(payload)
        ):
            with self.assertRaisesRegex(RuntimeError, "仅支持名片模板"):
                ocr_engine._call_glm("image", invoice_prompt)

    def test_business_card_layout_response_still_uses_local_mapping(self):
        payload = {"md_results": ["张三", "某某有限公司", "销售经理", "13800138000"]}
        business_card_prompt = "名片 字段列表：姓名, 公司, 职位, 手机, 座机, 邮箱, 地址"

        with patch.object(ocr_engine, "_compress_base64_image", return_value="image"), patch(
            "ocr_engine.requests.post", return_value=_FakeResponse(payload)
        ):
            result = ocr_engine._call_glm("image", business_card_prompt)

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "某某有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")


if __name__ == "__main__":
    unittest.main()
