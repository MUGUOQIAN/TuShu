import unittest

from ocr_engine import _extract_text_chunks, _map_business_card_fields


class GlmLayoutParsingTests(unittest.TestCase):
    def test_extracts_text_from_markdown_string_md_results(self):
        response = {
            "md_results": "\n".join(
                [
                    "# 上海图书科技有限公司",
                    "张三",
                    "销售经理",
                    "手机 13812345678",
                    "邮箱 zhangsan@example.com",
                    "地址 上海市浦东新区世纪大道100号",
                ]
            )
        }

        chunks = _extract_text_chunks(response)
        result = _map_business_card_fields(chunks)

        self.assertIn("张三", chunks)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海图书科技有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")
        self.assertEqual(result["地址"], "地址 上海市浦东新区世纪大道100号")


if __name__ == "__main__":
    unittest.main()
