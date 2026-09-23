import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardCompanyLabelTest(unittest.TestCase):
    def test_company_phone_line_does_not_replace_real_company(self):
        result = _map_business_card_fields(
            [
                "公司电话：021-12345678",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "上海市南京路1号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["座机"], "021-12345678")
        self.assertEqual(result["地址"], "上海市南京路1号")

    def test_company_address_line_does_not_replace_real_company(self):
        result = _map_business_card_fields(
            [
                "公司地址：上海市南京路100号",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
            ]
        )
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "上海市南京路100号")
        self.assertEqual(result["姓名"], "张三")

    def test_company_email_and_website_are_not_company_names(self):
        result = _map_business_card_fields(
            [
                "公司邮箱：hr@example.com",
                "公司网址：www.example.com",
                "李四",
                "北京星河集团",
                "13900001111",
            ]
        )
        self.assertEqual(result["公司"], "北京星河集团")
        self.assertEqual(result["邮箱"], "hr@example.com")
        self.assertEqual(result["姓名"], "李四")

    def test_address_label_only_leaves_company_empty(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "公司地址：上海市南京路100号",
                "13800138000",
            ]
        )
        self.assertEqual(result["公司"], "")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_company_name_containing_phone_words_is_kept(self):
        result = _map_business_card_fields(
            [
                "公司电话技术服务有限公司",
                "王五",
                "13800138000",
            ]
        )
        self.assertEqual(result["公司"], "公司电话技术服务有限公司")
        self.assertEqual(result["姓名"], "王五")

    def test_plain_business_card_still_maps(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "zhangsan@example.com",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")


if __name__ == "__main__":
    unittest.main()
