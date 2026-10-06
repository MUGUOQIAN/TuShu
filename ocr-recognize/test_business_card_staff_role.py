import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardStaffRoleTest(unittest.TestCase):
    def test_staff_role_line_does_not_replace_name(self):
        samples = ["内勤", "外勤", "督导", "领班"]
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

    def test_qualified_staff_title_is_not_truncated_into_name(self):
        cases = [
            "高级内勤",
            "资深外勤",
            "现场督导",
            "质量督导",
            "楼层领班",
            "值班领班",
            "行政内勤",
            "现场质量督导",
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
            self.assertNotIn(
                result["姓名"],
                ("高级内勤", "资深外勤", "现场督导", "现场质量", "华东区内"),
                title,
            )
            self.assertEqual(result["地址"], "南京西路100号", title)
            self.assertEqual(result["公司"], "上海云启科技有限公司", title)

    def test_staff_role_and_name_on_same_line(self):
        cases = [
            ("内勤张三", "内勤", "张三"),
            ("内勤 张三", "内勤", "张三"),
            ("内勤：张三", "内勤", "张三"),
            ("张三领班", "领班", "张三"),
            ("张三 外勤", "外勤", "张三"),
            ("督导李娜", "督导", "李娜"),
            ("督导：王芳", "督导", "王芳"),
            ("高级内勤王芳", "高级内勤", "王芳"),
            ("楼层领班：李四", "楼层领班", "李四"),
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
            self.assertNotIn(result["姓名"], ("内勤张三", "张三领班", "督导李娜", "高级内勤"), line)

    def test_staff_role_does_not_override_manager_title(self):
        result = _map_business_card_fields(
            [
                "内勤",
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

    def test_sales_staff_line_still_uses_existing_title_rule(self):
        result = _map_business_card_fields(
            [
                "销售内勤",
                "张三",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售内勤")
        self.assertNotEqual(result["姓名"], "销售内勤")

    def test_company_containing_role_word_keeps_person(self):
        result = _map_business_card_fields(
            [
                "内勤科技有限公司",
                "李四",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["公司"], "内勤科技有限公司")
        self.assertEqual(result["职位"], "")
        self.assertEqual(result["地址"], "南京西路100号")

        service = _map_business_card_fields(
            [
                "领班服务有限公司",
                "王芳",
                "13900139000",
                "南京西路100号",
            ]
        )
        self.assertEqual(service["姓名"], "王芳")
        self.assertEqual(service["公司"], "领班服务有限公司")
        self.assertEqual(service["职位"], "")

    def test_staff_role_without_person_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "高级内勤",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["职位"], "高级内勤")
        self.assertNotEqual(result["姓名"], "高级内勤")
        self.assertNotEqual(result["姓名"], "内勤")

    def test_region_staff_line_does_not_steal_address(self):
        result = _map_business_card_fields(
            [
                "华东区内勤",
                "张三",
                "上海云启科技有限公司",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "华东区内勤")
        self.assertNotEqual(result["姓名"], "华东区内")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "南京西路100号")
        self.assertNotEqual(result["地址"], "华东区内勤")

    def test_office_suffix_is_not_a_person(self):
        result = _map_business_card_fields(
            [
                "内勤办公室",
                "张三",
                "上海某某科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertNotEqual(result["姓名"], "内勤办公")
        self.assertNotEqual(result["姓名"], "办公室")
        self.assertEqual(result["地址"], "南京西路100号")
        self.assertEqual(result["公司"], "上海某某科技有限公司")

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

        other = _map_business_card_fields(
            [
                "李娜",
                "中国工商银行股份有限公司",
                "13700137000",
                "南京西路100号",
            ]
        )
        self.assertEqual(other["姓名"], "李娜")
        self.assertEqual(other["公司"], "中国工商银行股份有限公司")
        self.assertEqual(other["职位"], "")
        self.assertEqual(other["地址"], "南京西路100号")

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
