import unittest

from ocr_engine import _map_business_card_fields


class MobileLabelPriorityTest(unittest.TestCase):
    def test_labeled_mobile_wins_over_earlier_phone(self):
        result = _map_business_card_fields(
            [
                "上海示例科技有限公司",
                "电话：13900001111",
                "张三",
                "销售经理",
                "手机：13812345678",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["座机"], "")

    def test_mobile_label_with_country_code_wins(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "示例有限公司",
                "Tel: 13900001111",
                "Mobile: 86-13812345678",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")

    def test_m_label_wins_over_earlier_tel(self):
        result = _map_business_card_fields(
            [
                "张三",
                "示例有限公司",
                "电话：13900001111",
                "M: 13812345678",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")

    def test_hotline_does_not_replace_personal_mobile(self):
        result = _map_business_card_fields(
            [
                "销售热线：13700001111",
                "张三",
                "手机：13812345678",
                "示例有限公司",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")

    def test_company_mobile_does_not_beat_personal_mobile(self):
        result = _map_business_card_fields(
            [
                "张三",
                "公司手机：13900001111",
                "手机：13812345678",
                "示例有限公司",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")

    def test_company_mobile_kept_when_it_is_the_only_number(self):
        result = _map_business_card_fields(
            [
                "张三",
                "公司手机：13900001111",
                "示例有限公司",
            ]
        )
        self.assertEqual(result["手机"], "13900001111")

    def test_single_unlabeled_mobile_still_kept(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "示例有限公司",
                "13812345678",
                "021-60542168",
                "sales@example.com",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["座机"], "021-60542168")
        self.assertEqual(result["邮箱"], "sales@example.com")

    def test_only_general_phone_is_kept(self):
        result = _map_business_card_fields(
            [
                "张三",
                "电话：13812345678",
                "示例有限公司",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")

    def test_company_name_on_previous_line_does_not_steal_label(self):
        result = _map_business_card_fields(
            [
                "手机配件有限公司",
                "13900001111",
                "张三",
                "手机：13812345678",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")

    def test_same_line_company_name_does_not_count_as_label(self):
        result = _map_business_card_fields(
            [
                "手机配件有限公司 13900001111",
                "张三",
                "手机号码：13812345678",
            ]
        )
        self.assertEqual(result["手机"], "13812345678")


if __name__ == "__main__":
    unittest.main()
