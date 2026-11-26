import unittest

from ocr_engine import _map_business_card_fields


class AddressExtractionTest(unittest.TestCase):
    def test_split_chinese_address_keeps_street(self):
        result = _map_business_card_fields(
            [
                "张伟",
                "销售经理",
                "上海久协机械有限公司",
                "上海市浦东新区",
                "世纪大道100号国贸大厦18楼",
                "13800138000",
                "zhangwei@jiuxie.com",
            ]
        )
        self.assertEqual(result["姓名"], "张伟")
        self.assertEqual(result["公司"], "上海久协机械有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号国贸大厦18楼")

    def test_district_road_and_building_on_separate_lines(self):
        result = _map_business_card_fields(
            [
                "李娜",
                "上海市浦东新区",
                "世纪大道",
                "100号国贸大厦18楼",
            ]
        )
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号国贸大厦18楼")

    def test_complete_chinese_address_does_not_absorb_english_translation(self):
        result = _map_business_card_fields(
            [
                "赵社民",
                "Shermin Zhao",
                "总经理",
                "General Manager",
                "上海久协机械设备有限公司",
                "Shanghai Jiuxie Machinery Co.,Ltd.",
                "工厂地址：上海浦东宣秋路466号2栋",
                "Factory Add: Building No.2, Xuanqiu Road, Pudong, Shanghai, China",
                "Tel: 86-21-60542168",
                "Fax: 86-21-50521415",
                "Mobile: 86-13120543237",
                "Email: sales@jxprecisionparts.com",
            ]
        )
        self.assertEqual(result["姓名"], "赵社民")
        self.assertEqual(result["公司"], "上海久协机械设备有限公司")
        self.assertEqual(result["职位"], "总经理")
        self.assertEqual(result["手机"], "13120543237")
        self.assertEqual(result["座机"], "")
        self.assertEqual(result["邮箱"], "sales@jxprecisionparts.com")
        self.assertEqual(result["地址"], "工厂地址：上海浦东宣秋路466号2栋")

    def test_english_street_without_address_label(self):
        result = _map_business_card_fields(
            [
                "Jane Doe",
                "Sales Manager",
                "Acme Inc",
                "Building No.2, Xuanqiu Road, Pudong, Shanghai",
                "Mobile: 13812345678",
                "jane@acme.com",
            ]
        )
        self.assertEqual(result["姓名"], "Jane Doe")
        self.assertEqual(result["公司"], "Acme Inc")
        self.assertEqual(result["职位"], "Sales Manager")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["地址"], "Building No.2, Xuanqiu Road, Pudong, Shanghai")

    def test_chinese_address_preferred_when_english_line_comes_first(self):
        result = _map_business_card_fields(
            [
                "Factory Add: Building No.2, Xuanqiu Road",
                "工厂地址：上海浦东宣秋路466号2栋",
            ]
        )
        self.assertEqual(result["地址"], "工厂地址：上海浦东宣秋路466号2栋")

    def test_split_english_label_keeps_following_street(self):
        result = _map_business_card_fields(
            [
                "Jane Doe",
                "Factory Add:",
                "Building No.2, Xuanqiu Road",
            ]
        )
        self.assertEqual(result["地址"], "Factory Add: Building No.2, Xuanqiu Road")

    def test_contact_line_between_district_and_street_is_not_kept(self):
        result = _map_business_card_fields(
            [
                "张伟",
                "上海市浦东新区",
                "手机：13800138000",
                "世纪大道8号",
            ]
        )
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道8号")
        self.assertNotIn("13800138000", result["地址"])
        self.assertEqual(result["手机"], "13800138000")


if __name__ == "__main__":
    unittest.main()
