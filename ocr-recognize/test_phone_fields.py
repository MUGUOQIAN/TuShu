import unittest

from ocr_engine import _map_business_card_fields


class PhoneFieldExtractionTest(unittest.TestCase):
    def test_repo_sample_card_keeps_mobile_and_office_number(self):
        # tushu/test/22222.jpg 上的印刷格式是 86-13120543237 / 86-21-60542168。
        # 传真行不应盖过前面的 Tel。
        result = _map_business_card_fields(
            [
                "赵社民 Shermin Zhao",
                "总经理",
                "General Manager",
                "上海久协机械设备有限公司",
                "Shanghai Jiuxie Machinery Co., Ltd.",
                "Mobile: 86-13120543237",
                "Tel: 86-21-60542168",
                "Fax: 86-21-50521415",
                "Email: sales@jxprecisionparts.com",
                "http://weibo.com/3214361603",
                "工厂地址: 上海市宣秋路466号2栋",
            ]
        )
        self.assertEqual(result["姓名"], "赵社民 (Shermin Zhao)")
        self.assertEqual(result["公司"], "上海久协机械设备有限公司")
        self.assertEqual(result["职位"], "总经理")
        self.assertEqual(result["手机"], "13120543237")
        self.assertEqual(result["座机"], "021-60542168")
        self.assertEqual(result["邮箱"], "sales@jxprecisionparts.com")

    def test_grouped_mobile_and_landline_are_not_dropped(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "某某科技有限公司",
                "手机: 138-1234-5678",
                "电话: 021-1234-5678",
                "zhangsan@example.com",
                "上海市浦东新区张江路88号",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["座机"], "021-12345678")
        self.assertEqual(result["姓名"], "张三")

    def test_spaced_country_code_and_parenthesized_area(self):
        result = _map_business_card_fields(
            [
                "李四",
                "产品经理",
                "北京示例技术有限公司",
                "M: +86 138 1234 5678",
                "T: (010) 6234-5678",
                "li@example.com",
                "北京市海淀区中关村大街1号",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["座机"], "010-62345678")

    def test_contiguous_numbers_stay_canonical(self):
        result = _map_business_card_fields(
            [
                "王五",
                "工程师",
                "杭州示例有限公司",
                "13812345678",
                "0571-87654321",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["座机"], "0571-87654321")

    def test_fullwidth_digits(self):
        result = _map_business_card_fields(
            [
                "赵六",
                "主管",
                "深圳示例有限公司",
                "手机：１３８１２３４５６７８",
                "电话：０２１－６０５４２１６８",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["座机"], "021-60542168")

    def test_newline_does_not_glue_unrelated_numbers(self):
        result = _map_business_card_fields(
            [
                "钱七",
                "顾问",
                "南京示例集团",
                "分机 138",
                "订单 12345678",
            ]
        )
        self.assertEqual(result["手机"], "")
        self.assertEqual(result["座机"], "")

    def test_office_number_wins_when_fax_is_listed_first(self):
        result = _map_business_card_fields(
            [
                "张三",
                "经理",
                "示例有限公司",
                "Fax: 86-21-50521415",
                "Tel: 86-21-60542168",
            ]
        )
        self.assertEqual(result["座机"], "021-60542168")

    def test_only_fax_is_kept_when_no_office_number(self):
        result = _map_business_card_fields(
            [
                "孙八",
                "主任",
                "苏州示例有限公司",
                "传真: 86-512-65432109",
            ]
        )
        self.assertEqual(result["座机"], "0512-65432109")

    def test_service_number_and_address_are_not_phones(self):
        result = _map_business_card_fields(
            [
                "周九",
                "销售",
                "成都示例有限公司",
                "客服 400-800-1234",
                "成都市武侯区人民南路88号",
            ]
        )
        self.assertEqual(result["手机"], "")
        self.assertEqual(result["座机"], "")


if __name__ == "__main__":
    unittest.main()
