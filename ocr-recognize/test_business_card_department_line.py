import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardDepartmentLineTest(unittest.TestCase):
    def test_hr_department_does_not_replace_name(self):
        result = _map_business_card_fields(
            [
                "人力资源部",
                "张三",
                "招聘主管",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "招聘主管")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_long_department_names_do_not_replace_person(self):
        samples = [
            ("信息技术部", "王五", "工程师"),
            ("财务管理部", "赵六", "财务经理"),
            ("公共关系部", "钱七", "公关经理"),
            ("质量管理部", "孙八", "质量经理"),
            ("客户服务部", "李四", "客服经理"),
            ("市场营销部", "李四", "产品经理"),
        ]
        for department, person, title in samples:
            result = _map_business_card_fields(
                [
                    department,
                    person,
                    title,
                    "上海云启科技有限公司",
                    "南京西路100号",
                ]
            )
            self.assertEqual(result["姓名"], person, department)
            self.assertEqual(result["公司"], "上海云启科技有限公司", department)
            self.assertEqual(result["职位"], title, department)
            self.assertEqual(result["地址"], "南京西路100号", department)

    def test_short_department_does_not_replace_address_or_name(self):
        result = _map_business_card_fields(
            [
                "市场部",
                "张三",
                "产品经理",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "产品经理")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_sales_department_does_not_replace_title(self):
        result = _map_business_card_fields(
            [
                "销售部",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_department_and_name_on_same_line(self):
        result = _map_business_card_fields(
            [
                "人力资源部 张三",
                "招聘主管",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_name_before_department_is_kept(self):
        result = _map_business_card_fields(
            [
                "张三",
                "人力资源部",
                "招聘主管",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "招聘主管")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_department_label_with_street_keeps_address(self):
        result = _map_business_card_fields(
            [
                "市场部：南京西路100号",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["地址"], "市场部：南京西路100号")

    def test_plain_names_and_company_still_map(self):
        result = _map_business_card_fields(
            [
                "欧阳娜娜",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "欧阳娜娜")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")

        two_char = _map_business_card_fields(
            ["李明", "杭州云启科技有限公司", "文三路88号"]
        )
        self.assertEqual(two_char["姓名"], "李明")
        self.assertEqual(two_char["公司"], "杭州云启科技有限公司")
        self.assertEqual(two_char["地址"], "文三路88号")


if __name__ == "__main__":
    unittest.main()
