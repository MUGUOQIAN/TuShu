import unittest

import ocr_engine


class OcrEngineFieldMappingTest(unittest.TestCase):
    def test_invoice_layout_text_maps_invoice_fields(self):
        data = {
            "md_results": [
                "发票号码: 031002300111\n"
                "开票日期: 2026年05月01日\n"
                "购买方名称: 上海示例科技有限公司\n"
                "销售方名称: 北京供应商有限公司\n"
                "价税合计: ¥1234.56\n"
                "税额: ¥123.45"
            ]
        }

        chunks = ocr_engine._extract_text_chunks(data)
        result = ocr_engine._map_layout_text_fields(
            chunks,
            template_type="invoice",
            expected_fields=["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )

        self.assertEqual(result["发票号码"], "031002300111")
        self.assertEqual(result["开票日期"], "2026年05月01日")
        self.assertEqual(result["购买方名称"], "上海示例科技有限公司")
        self.assertEqual(result["销售方名称"], "北京供应商有限公司")
        self.assertEqual(result["价税合计金额"], "¥1234.56")
        self.assertEqual(result["税额"], "¥123.45")

    def test_custom_layout_text_maps_requested_fields(self):
        data = {
            "layout_details": [
                [
                    {"content": "姓名: 张三"},
                    {"content": "电话: 13800138000"},
                    {"content": "地址: 上海市浦东新区示例路1号"},
                ]
            ]
        }

        chunks = ocr_engine._extract_text_chunks(data)
        result = ocr_engine._map_layout_text_fields(
            chunks,
            template_type="custom",
            expected_fields=["姓名", "电话", "地址"],
        )

        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["电话"], "13800138000")
        self.assertEqual(result["地址"], "上海市浦东新区示例路1号")


if __name__ == "__main__":
    unittest.main()
