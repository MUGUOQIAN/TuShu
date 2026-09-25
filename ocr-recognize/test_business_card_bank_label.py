import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardBankLabelTest(unittest.TestCase):
    def test_opening_bank_does_not_replace_name_or_company(self):
        result = _map_business_card_fields(
            [
                "开户银行：中国工商银行",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["地址"], "上海市南京路100号")
        self.assertEqual(result["手机"], "13800138000")

    def test_opening_bank_with_company_suffix_does_not_replace_company(self):
        result = _map_business_card_fields(
            [
                "开户银行：中国工商银行股份有限公司",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_receiving_and_settlement_bank_do_not_replace_name(self):
        receiving = _map_business_card_fields(
            [
                "收款银行：招商银行上海分行",
                "李四",
                "北京星河集团",
                "北京市朝阳区建国路88号",
            ]
        )
        self.assertEqual(receiving["姓名"], "李四")
        self.assertEqual(receiving["公司"], "北京星河集团")
        self.assertEqual(receiving["地址"], "北京市朝阳区建国路88号")

        settlement = _map_business_card_fields(
            [
                "结算银行：中国银行",
                "汇款银行：建设银行",
                "王五",
                "杭州云启科技有限公司",
                "杭州市文三路1号",
            ]
        )
        self.assertEqual(settlement["姓名"], "王五")
        self.assertEqual(settlement["公司"], "杭州云启科技有限公司")
        self.assertEqual(settlement["地址"], "杭州市文三路1号")

    def test_bank_account_lines_do_not_replace_address_or_name(self):
        result = _map_business_card_fields(
            [
                "银行账号：6222021234567890",
                "银行卡号：6222021234567890",
                "银行账户：6222021234567890",
                "收款账号：6222021234567890123",
                "基本户账号：6222021234567890",
                "对公账号：6222021234567890",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_account_number_without_separator_is_not_an_address(self):
        result = _map_business_card_fields(
            [
                "银行账号6222021234567890",
                "张三",
                "上海云启科技有限公司",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")

    def test_company_name_containing_bank_words_is_kept(self):
        result = _map_business_card_fields(
            [
                "招商银行股份有限公司",
                "张三",
                "销售经理",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["公司"], "招商银行股份有限公司")
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["地址"], "上海市南京路100号")

        technical = _map_business_card_fields(
            [
                "开户银行技术服务有限公司",
                "王五",
                "13800138000",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(technical["公司"], "开户银行技术服务有限公司")
        self.assertEqual(technical["姓名"], "王五")
        self.assertEqual(technical["地址"], "上海市南京路100号")

    def test_bank_label_alone_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "开户银行：中国工商银行",
                "上海云启科技有限公司",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_plain_business_card_still_maps(self):
        result = _map_business_card_fields(
            [
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "13800138000",
                "zhangsan@example.com",
                "上海市浦东新区世纪大道100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["邮箱"], "zhangsan@example.com")
        self.assertEqual(result["地址"], "上海市浦东新区世纪大道100号")


if __name__ == "__main__":
    unittest.main()
