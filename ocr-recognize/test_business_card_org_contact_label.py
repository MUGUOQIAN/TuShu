import unittest

from ocr_engine import _map_business_card_fields


class BusinessCardOrgContactLabelTest(unittest.TestCase):
    def test_company_mobile_does_not_replace_company(self):
        result = _map_business_card_fields(
            [
                "公司手机：13800138000",
                "张三",
                "销售经理",
                "上海云启科技有限公司",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["手机"], "13800138000")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_group_phone_does_not_replace_company(self):
        result = _map_business_card_fields(
            [
                "集团电话：021-66668888",
                "张三",
                "上海云启科技有限公司",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["座机"], "021-66668888")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_group_address_keeps_company_and_street(self):
        result = _map_business_card_fields(
            [
                "集团地址：上海市南京路100号",
                "张三",
                "销售经理",
                "上海云启集团",
            ]
        )
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["公司"], "上海云启集团")
        self.assertEqual(result["职位"], "销售经理")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_branch_phone_and_group_fax_do_not_replace_company(self):
        branch = _map_business_card_fields(
            [
                "分公司电话：021-66668888",
                "李四",
                "杭州云启科技有限公司",
                "杭州市文三路1号",
            ]
        )
        self.assertEqual(branch["姓名"], "李四")
        self.assertEqual(branch["公司"], "杭州云启科技有限公司")
        self.assertEqual(branch["地址"], "杭州市文三路1号")

        fax = _map_business_card_fields(
            [
                "集团传真：021-58301234",
                "集团网址：example.cn",
                "王五",
                "北京星河集团",
                "北京市朝阳区建国路88号",
            ]
        )
        self.assertEqual(fax["姓名"], "王五")
        self.assertEqual(fax["公司"], "北京星河集团")
        self.assertEqual(fax["地址"], "北京市朝阳区建国路88号")

    def test_branch_phone_fragment_is_not_a_name(self):
        result = _map_business_card_fields(
            [
                "分公司电话：021-66668888",
                "上海云启科技有限公司",
                "销售经理",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["姓名"], "")
        self.assertEqual(result["公司"], "上海云启科技有限公司")
        self.assertEqual(result["地址"], "上海市南京路100号")

    def test_company_name_containing_org_contact_words_is_kept(self):
        result = _map_business_card_fields(
            [
                "集团电话技术服务有限公司",
                "张三",
                "销售经理",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(result["公司"], "集团电话技术服务有限公司")
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["地址"], "上海市南京路100号")

        mobile_company = _map_business_card_fields(
            [
                "公司手机科技有限公司",
                "王五",
                "13800138000",
                "上海市南京路100号",
            ]
        )
        self.assertEqual(mobile_company["公司"], "公司手机科技有限公司")
        self.assertEqual(mobile_company["姓名"], "王五")

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
