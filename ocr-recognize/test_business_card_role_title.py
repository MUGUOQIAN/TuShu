import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardRoleTitleTest(unittest.TestCase):
    def test_role_title_line_does_not_replace_name(self):
        samples = ["管理员", "柜员", "信贷员"]
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

    def test_qualified_admin_title_is_not_truncated_into_name(self):
        cases = [
            "系统管理员",
            "网络管理员",
            "数据库管理员",
            "高级管理员",
            "资深系统管理员",
            "综合柜员",
            "对公信贷员",
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
            self.assertNotIn(result["姓名"], ("系统管理", "网络管理", "数据库管", "高级管理", "资深系统"), title)
            self.assertEqual(result["地址"], "南京西路100号", title)
            self.assertEqual(result["公司"], "上海云启科技有限公司", title)

    def test_role_title_and_name_on_same_line(self):
        cases = [
            ("管理员张三", "管理员", "张三"),
            ("管理员 张三", "管理员", "张三"),
            ("管理员：张三", "管理员", "张三"),
            ("张三管理员", "管理员", "张三"),
            ("张三 信贷员", "信贷员", "张三"),
            ("柜员李娜", "柜员", "李娜"),
            ("系统管理员王芳", "系统管理员", "王芳"),
            ("综合柜员：李四", "综合柜员", "李四"),
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
            self.assertNotIn(result["姓名"], ("管理员张", "张三管理", "柜员李娜", "系统管理"), line)

    def test_role_title_does_not_override_manager_title(self):
        result = _map_business_card_fields(
            [
                "系统管理员",
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

    def test_sales_admin_line_still_uses_existing_title_rule(self):
        result = _map_business_card_fields(
            [
                "销售管理员",
                "张三",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售管理员")
        self.assertNotEqual(result["姓名"], "销售管理")

    def test_company_containing_role_word_keeps_person(self):
        result = _map_business_card_fields(
            [
                "管理员科技有限公司",
                "李四",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["公司"], "管理员科技有限公司")
        self.assertEqual(result["职位"], "")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_role_title_without_person_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "系统管理员",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["职位"], "系统管理员")
        self.assertNotEqual(result["姓名"], "系统管理")
        self.assertNotEqual(result["姓名"], "管理员")

    def test_region_admin_line_does_not_steal_address(self):
        result = _map_business_card_fields(
            [
                "华东区管理员",
                "张三",
                "上海云启科技有限公司",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "华东区管理员")
        self.assertNotEqual(result["姓名"], "华东区管")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "南京西路100号")
        self.assertNotEqual(result["地址"], "华东区管理员")

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

    def test_two_char_name_and_guan_surname_unchanged(self):
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

        guan = _map_business_card_fields(
            [
                "管伟",
                "中国工商银行股份有限公司",
                "13700137000",
                "南京西路100号",
            ]
        )
        self.assertEqual(guan["姓名"], "管伟")
        self.assertEqual(guan["公司"], "中国工商银行股份有限公司")
        self.assertEqual(guan["职位"], "")
        self.assertEqual(guan["地址"], "南京西路100号")

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
