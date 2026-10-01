import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardGovernanceTitleTest(unittest.TestCase):
    def test_supervisor_line_does_not_replace_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "监事",
                "王芳",
                "13800138000",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "王芳")
        self.assertEqual(result["职位"], "监事")
        self.assertEqual(result["公司"], "上海某某科技有限公司")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")

    def test_governance_titles_do_not_replace_person(self):
        samples = [
            "监事长",
            "董事",
            "独立董事",
            "执行董事",
            "常务董事",
            "非执行董事",
            "董事长",
            "名誉董事长",
            "会长",
            "副会长",
            "名誉会长",
            "理事",
            "常务理事",
            "理事长",
            "副理事长",
            "委员",
            "院士",
            "股东",
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
            ("监事 王芳", "监事", "王芳"),
            ("会长：张三", "会长", "张三"),
            ("王芳 监事", "监事", "王芳"),
            ("监事王芳", "监事", "王芳"),
            ("独立董事 李四", "独立董事", "李四"),
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

    def test_governance_line_does_not_override_job_title(self):
        result = _map_business_card_fields(
            [
                "监事",
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

    def test_organization_suffix_is_not_a_person(self):
        result = _map_business_card_fields(
            [
                "院士工作室",
                "李四",
                "上海某某科技有限公司",
                "13700137000",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertNotEqual(result["姓名"], "工作室")
        self.assertNotEqual(result["姓名"], "院士")
