import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardStaffTitleTest(unittest.TestCase):
    def test_staff_title_line_does_not_replace_name(self):
        samples = ["业务员", "技术员", "办事员", "操作员"]
        for title in samples:
            result = _map_business_card_fields(
                [
                    "上海某某科技有限公司",
                    title,
                    "张三",
                    "13800138000",
                    "南京西路100号",
                ]
            )
            self.assertEqual(result["姓名"], "张三", title)
            self.assertEqual(result["职位"], title, title)
            self.assertEqual(result["公司"], "上海某某科技有限公司", title)
            self.assertEqual(result["手机"], "13800138000", title)
            self.assertEqual(result["地址"], "南京西路100号", title)
            self.assertNotEqual(result["姓名"], title, title)

    def test_modified_staff_title_is_not_truncated_into_name(self):
        cases = [
            "高级业务员",
            "资深技术员",
            "见习办事员",
            "初级操作员",
            "中级技术员",
            "助理业务员",
            "实习办事员",
            "华东业务员",
        ]
        for title in cases:
            result = _map_business_card_fields(
                [
                    "上海云启科技有限公司",
                    title,
                    "王芳",
                    "13900139000",
                    "南京西路100号",
                ]
            )
            self.assertEqual(result["姓名"], "王芳", title)
            self.assertEqual(result["职位"], title, title)
            self.assertNotEqual(result["姓名"], "高级业务", title)
            self.assertNotEqual(result["姓名"], "资深技术", title)
            self.assertEqual(result["地址"], "南京西路100号", title)

    def test_staff_title_and_name_on_same_line(self):
        cases = [
            ("业务员张三", "业务员", "张三"),
            ("业务员 张三", "业务员", "张三"),
            ("业务员：张三", "业务员", "张三"),
            ("张三业务员", "业务员", "张三"),
            ("张三 操作员", "操作员", "张三"),
            ("技术员王芳", "技术员", "王芳"),
            ("资深技术员王芳", "资深技术员", "王芳"),
            ("高级业务员：李四", "高级业务员", "李四"),
        ]
        for line, title, person in cases:
            result = _map_business_card_fields(
                [
                    "上海某某科技有限公司",
                    line,
                    "13800138000",
                    "南京西路100号",
                ]
            )
            self.assertEqual(result["姓名"], person, line)
            self.assertEqual(result["职位"], title, line)
            self.assertEqual(result["公司"], "上海某某科技有限公司", line)
            self.assertNotIn(result["姓名"], ("业务员张", "张三业务", "资深技术"), line)

    def test_staff_title_does_not_override_manager_title(self):
        result = _map_business_card_fields(
            [
                "业务员",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["公司"], "上海云启科技有限公司")

    def test_sales_staff_line_still_uses_existing_title_rule(self):
        result = _map_business_card_fields(
            [
                "销售业务员",
                "张三",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售业务员")
        self.assertNotEqual(result["姓名"], "销售业务")

    def test_business_manager_unchanged(self):
        result = _map_business_card_fields(
            [
                "业务经理",
                "张三",
                "上海云启科技有限公司",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "业务经理")
        self.assertEqual(result["手机"], "13800138000")

    def test_company_containing_staff_word_keeps_person(self):
        result = _map_business_card_fields(
            [
                "业务员科技有限公司",
                "李四",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["公司"], "业务员科技有限公司")
        self.assertEqual(result["职位"], "")
        self.assertEqual(result["地址"], "南京西路100号")

        technician_company = _map_business_card_fields(
            [
                "技术员有限公司",
                "李四",
                "13700137000",
                "文三路1号",
            ]
        )
        self.assertEqual(technician_company["姓名"], "李四")
        self.assertEqual(technician_company["公司"], "技术员有限公司")
        self.assertEqual(technician_company["职位"], "")
        self.assertEqual(technician_company["地址"], "文三路1号")

    def test_staff_title_without_person_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "高级业务员",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["职位"], "高级业务员")
        self.assertNotEqual(result["姓名"], "高级业务")
        self.assertNotEqual(result["姓名"], "业务员")

    def test_region_staff_line_does_not_become_name(self):
        result = _map_business_card_fields(
            [
                "华东区业务员",
                "张三",
                "上海云启科技有限公司",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "华东区业务员")
        self.assertNotEqual(result["姓名"], "华东业务")
        self.assertNotEqual(result["姓名"], "华东区业")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_ordinary_card_unchanged(self):
        result = _map_business_card_fields(
            [
                "李明",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "liming@example.com",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "李明")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "liming@example.com")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_two_char_name_unchanged(self):
        result = _map_business_card_fields(
            [
                "马超",
                "上海云启科技有限公司",
                "13600136000",
            ]
        )
        self.assertEqual(result["姓名"], "马超")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "")

    def test_bilingual_name_unchanged(self):
        result = _map_business_card_fields(
            [
                "赵美娜 Shermin Zhao",
                "销售经理",
                "上海某某科技有限公司",
                "13800138000",
            ]
        )
        self.assertEqual(result["姓名"], "赵美娜 (Shermin Zhao)")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["公司"], "上海某某科技有限公司")

    def test_chief_engineer_unchanged(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "总工程师",
                "张三",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "总工程师")
        self.assertEqual(result["地址"], "南京西路100号")


if __name__ == "__main__":
    unittest.main()
