import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardExtensionLabelTest(unittest.TestCase):
    def test_extension_number_does_not_replace_street_address(self):
        result = _map_business_card_fields(
            [
                "分机号：8001",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "021-12345678",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "上海市南京路100号")
        self.assertEqual(result["座机"], "021-12345678")
        self.assertEqual(result["手机"], "13800138000")

    def test_switchboard_and_direct_line_do_not_replace_address(self):
        result = _map_business_card_fields(
            [
                "总机号：021-66668888",
                "直线号：021-87654321",
                "李四",
                "北京星河集团",
                "北京市朝阳区建国路88号",
            ]
        )
        self.assertEqual(result["公司"], "北京星河集团")
        self.assertEqual(result["地址"], "北京市朝阳区建国路88号")
        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["座机"], "021-66668888")

    def test_extension_line_alone_is_not_an_address(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "分机号：8001",
            ]
        )
        self.assertEqual(result["地址"], "")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["姓名"], "张三")

    def test_company_name_containing_extension_words_is_kept(self):
        result = _map_business_card_fields(
            [
                "直线号技术服务有限公司",
                "王五",
                "13800138000",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["公司"], "直线号技术服务有限公司")
        self.assertEqual(result["姓名"], "王五")
        # 「号」后面紧跟汉字，不是分机标签；含「号」的公司名仍可占住地址，不能误跳过。
        self.assertEqual(result["地址"], "直线号技术服务有限公司")

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
