import unittest

from ocr_engine import _map_business_card_fields


class TitleLineNameTests(unittest.TestCase):
    def _name(self, chunks):
        return _map_business_card_fields(chunks)["姓名"]

    def test_name_on_title_line_is_not_replaced_by_company(self):
        cases = [
            (["久协机械设备有限公司", "张三 总经理", "13812345678", "上海市浦东新区世纪大道100号"], "张三"),
            (["上海久协机械设备有限公司", "张三 总经理", "13812345678", "上海市南京路1号"], "张三"),
            (["上海久协机械设备有限公司", "总经理张三", "13812345678", "南京西路100号"], "张三"),
            (["上海久协机械设备有限公司", "张三销售经理", "13812345678", "南京西路100号"], "张三"),
            (["上海久协机械设备有限公司", "销售经理：张三", "13812345678", "南京西路100号"], "张三"),
            (["上海久协机械设备有限公司", "张三 | 销售总监", "13812345678"], "张三"),
            (["上海久协机械设备有限公司", "技术总监：王芳", "13812345678"], "王芳"),
            (["上海久协机械设备有限公司", "高级工程师张伟", "13812345678"], "张伟"),
            (["上海久协机械设备有限公司", "客户经理：李娜", "13812345678"], "李娜"),
            (["上海久协机械设备有限公司", "法律顾问张三", "13812345678"], "张三"),
            (["上海久协机械设备有限公司", "部门经理张三", "13812345678"], "张三"),
            (["上海久协机械设备有限公司", "客服经理：李娜", "13812345678"], "李娜"),
            (["上海久协机械设备有限公司", "欧阳娜娜 总经理", "13812345678"], "欧阳娜娜"),
            (["上海久协机械设备有限公司", "李总 总经理", "13812345678"], "李总"),
            (["上海久协机械设备有限公司", "副总经理张三", "13812345678"], "张三"),
            (["久协机械设备有限公司", "John Smith, General Manager", "13812345678"], "John Smith"),
            (["久协机械设备有限公司", "Nicole Brown, Manager", "13812345678"], "Nicole Brown"),
            (["上海久协机械设备有限公司", "赵社民 Shemin Zhao 总经理", "13812345678"], "赵社民 (Shemin Zhao)"),
            (["久协机械设备有限公司", "Mr John Smith, Director", "13812345678"], "John Smith"),
        ]
        for chunks, expected in cases:
            with self.subTest(chunks=chunks):
                self.assertEqual(self._name(chunks), expected)

    def test_clean_name_line_still_wins(self):
        self.assertEqual(
            self._name(["上海久协机械设备有限公司", "总经理", "张三", "13812345678", "南京西路100号"]),
            "张三",
        )
        self.assertEqual(
            self._name(["赵社民 Shemin Zhao", "总经理", "上海久协机械设备有限公司"]),
            "赵社民 (Shemin Zhao)",
        )
        self.assertEqual(
            self._name(["久协机械销售经理", "张三", "13812345678"]),
            "张三",
        )

    def test_title_only_line_does_not_invent_a_person(self):
        chunks = ["上海久协机械设备有限公司", "总经理", "13812345678", "上海市南京路1号"]
        result = _map_business_card_fields(chunks)
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["职位"], "总经理")
        self.assertEqual(result["公司"], "上海久协机械设备有限公司")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(self._name(["高级销售经理"]), "")
        self.assertEqual(self._name(["总经理"]), "")
        self.assertEqual(self._name(["久协机械销售经理"]), "")
        self.assertEqual(self._name(["销售总监"]), "")
        self.assertEqual(self._name(["上海久协机械设备有限公司", "销售热线"]), "")
        self.assertNotEqual(self._name(["久协机械设备有限公司", "销售热线"]), "热线")
        self.assertNotEqual(self._name(["销售微信"]), "微信")

    def test_company_line_is_not_parsed_as_title_name(self):
        chunks = ["张三销售有限公司", "13812345678"]
        result = _map_business_card_fields(chunks)
        self.assertEqual(result["公司"], "张三销售有限公司")
        self.assertNotEqual(result["姓名"], "张三")

    def test_other_fields_survive_title_line(self):
        chunks = [
            "久协机械设备有限公司",
            "张三 总经理",
            "13812345678",
            "021-60542168",
            "sales@example.com",
            "上海市浦东新区世纪大道100号",
        ]
        result = _map_business_card_fields(chunks)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["职位"], "张三 总经理")
        self.assertEqual(result["公司"], "久协机械设备有限公司")
        self.assertEqual(result["手机"], "13812345678")
        self.assertEqual(result["座机"], "021-60542168")
        self.assertEqual(result["邮箱"], "sales@example.com")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")


if __name__ == "__main__":
    unittest.main()
