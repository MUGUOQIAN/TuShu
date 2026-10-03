import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardCompanyFormTest(unittest.TestCase):
    def test_kabushiki_prefix_does_not_replace_name_or_drop_company(self):
        result = _map_business_card_fields(
            [
                "株式会社山田制作所",
                "佐藤健",
                "13800138000",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "佐藤健")
        self.assertEqual(result["公司"], "株式会社山田制作所")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")
        self.assertNotEqual(result["姓名"], "株式会社")

    def test_kabushiki_suffix_does_not_truncate_into_name(self):
        result = _map_business_card_fields(
            [
                "山田制作所株式会社",
                "佐藤健",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "佐藤健")
        self.assertEqual(result["公司"], "山田制作所株式会社")
        self.assertNotEqual(result["姓名"], "山田制作")
        self.assertEqual(result["地址"], "南京西路100号")

    def test_godo_kaisha_keeps_company_and_person(self):
        result = _map_business_card_fields(
            [
                "合同会社青空设计",
                "佐藤健",
                "13900139000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "佐藤健")
        self.assertEqual(result["公司"], "合同会社青空设计")
        self.assertNotEqual(result["姓名"], "合同会社")

    def test_yugen_kaisha_still_keeps_person(self):
        result = _map_business_card_fields(
            [
                "有限会社佐藤商店",
                "王芳",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "王芳")
        self.assertEqual(result["公司"], "有限会社佐藤商店")
        self.assertEqual(result["职位"], "")

    def test_company_line_alone_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "株式会社山田制作所",
                "13800138000",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["公司"], "株式会社山田制作所")
        self.assertNotEqual(result["姓名"], "株式会社")

    def test_role_line_does_not_replace_name(self):
        samples = [
            "主席",
            "副主席",
            "名誉主席",
            "执行主席",
            "常务主席",
            "轮值主席",
            "社长",
            "副社长",
            "课长",
            "副课长",
            "次长",
            "系长",
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

    def test_japanese_card_with_title_keeps_person(self):
        result = _map_business_card_fields(
            [
                "株式会社山田制作所",
                "社长",
                "佐藤健",
                "13800138000",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "佐藤健")
        self.assertEqual(result["职位"], "社长")
        self.assertEqual(result["公司"], "株式会社山田制作所")
        self.assertEqual(result["手机"], "13800138000")

    def test_role_and_name_on_same_line(self):
        cases = [
            ("社长 佐藤健", "社长", "佐藤健"),
            ("社长：佐藤健", "社长", "佐藤健"),
            ("课长张三", "课长", "张三"),
            ("张三主席", "主席", "张三"),
            ("副主席 王芳", "副主席", "王芳"),
            ("名誉主席 李四", "名誉主席", "李四"),
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

    def test_role_does_not_override_manager_title(self):
        result = _map_business_card_fields(
            [
                "社长",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["公司"], "上海云启科技有限公司")

    def test_company_containing_role_word_keeps_person(self):
        result = _map_business_card_fields(
            [
                "社长科技有限公司",
                "李四",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertEqual(result["公司"], "社长科技有限公司")
        self.assertEqual(result["职位"], "")
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

    def test_role_organization_suffix_is_not_a_person(self):
        result = _map_business_card_fields(
            [
                "主席办公室",
                "李四",
                "上海某某科技有限公司",
                "13700137000",
            ]
        )
        self.assertEqual(result["姓名"], "李四")
        self.assertNotEqual(result["姓名"], "主席")
        self.assertNotEqual(result["姓名"], "主席办公")
        self.assertEqual(result["公司"], "上海某某科技有限公司")

    def test_role_without_person_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "上海某某科技有限公司",
                "名誉主席",
                "13800138000",
                "南京西路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["职位"], "名誉主席")
        self.assertNotEqual(result["姓名"], "名誉主席")
        self.assertNotEqual(result["姓名"], "名誉")
