import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardShortTitleTest(unittest.TestCase):
    def test_vice_gm_abbreviation_does_not_replace_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "副总",
                "张三",
                "13800138000",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "副总")
        self.assertEqual(result["公司"], "上海某某科技有限公司")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")

    def test_short_titles_do_not_replace_person(self):
        samples = [
            "常务副总",
            "执行副总",
            "副总工",
            "正高工",
            "总工",
            "高工",
            "助工",
            "董秘",
            "总助",
            "注会",
            "建筑师",
            "结构师",
            "咨询师",
            "经济师",
            "税务师",
            "理财师",
            "规划师",
            "评估师",
            "统计师",
            "专家",
            "技术专家",
            "高级经济师",
            "心理咨询师",
            "博士生导师",
            "指导教师",
            "教师",
            "导师",
            "博导",
        ]
        for title in samples:
            result = _map_business_card_fields(
                [
                    "上海某某科技有限公司",
                    title,
                    "张三",
                    "13900139000",
                    "南京西路100号",
                ]
            )
            self.assertEqual(result["姓名"], "张三", title)
            self.assertEqual(result["职位"], title, title)
            self.assertEqual(result["公司"], "上海某某科技有限公司", title)
            self.assertEqual(result["地址"], "南京西路100号", title)
            self.assertEqual(result["手机"], "13900139000", title)

    def test_title_and_name_on_same_line(self):
        cases = [
            ("副总 张三", "副总", "张三"),
            ("副总：张三", "副总", "张三"),
            ("高工张三", "高工", "张三"),
            ("张三董秘", "董秘", "张三"),
            ("建筑师 李四", "建筑师", "李四"),
            ("常务副总 王芳", "常务副总", "王芳"),
            ("张三 副总", "副总", "张三"),
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
            self.assertEqual(result["地址"], "南京西路100号", line)

    def test_short_title_does_not_override_job_title(self):
        result = _map_business_card_fields(
            [
                "副总",
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

    def test_existing_manager_titles_unchanged(self):
        for title in ("总工程师", "副总经理", "销售经理"):
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

    def test_organization_suffix_is_not_a_person(self):
        result = _map_business_card_fields(
            [
                "专家工作室",
                "李四",
                "上海某某科技有限公司",
                "13700137000",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertNotEqual(result["姓名"], "专家")
        self.assertNotEqual(result["姓名"], "专家工作")

        firm = _map_business_card_fields(
            [
                "建筑师事务所",
                "李四",
                "13700137000",
            ]
        )
        self.assertEqual(firm["姓名"], "李四")
        self.assertNotEqual(firm["姓名"], "建筑师")
        self.assertNotEqual(firm["姓名"], "建筑师事")

    def test_company_containing_abbreviation_keeps_person(self):
        result = _map_business_card_fields(
            [
                "副总科技有限公司",
                "李四",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["公司"], "副总科技有限公司")
        self.assertEqual(result["职位"], "")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_title_without_person_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "高级经济师",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["职位"], "高级经济师")
        self.assertNotEqual(result["姓名"], "高级经济")
        self.assertEqual(result["公司"], "上海某某科技有限公司")
        self.assertEqual(result["地址"], "南京西路100号")
