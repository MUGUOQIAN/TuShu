import base64
import io
import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import index
import ocr_engine
from config import MAX_IMAGE_SIZE
from prompt_templates import TEMPLATE_MAP


SMALL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/6X7qN8AAAAASUVORK5CYII="
)


def mock_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = json.dumps(payload, ensure_ascii=False)
    return response


def _jpeg_base64(width=20, height=20):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


class OcrEngineTest(unittest.TestCase):
    def setUp(self):
        self.api_key_patch = patch.object(ocr_engine, "GLM_API_KEY", "test-key")
        self.api_key_patch.start()

    def tearDown(self):
        self.api_key_patch.stop()

    def test_image_pixel_limit_is_checked_before_conversion(self):
        with self.assertRaisesRegex(ValueError, "图片像素尺寸过大"):
            ocr_engine._prepare_image_data_uri(SMALL_PNG_BASE64, max_pixels=0)

    def test_small_png_keeps_png_mime_type(self):
        data_uri = ocr_engine._prepare_image_data_uri(SMALL_PNG_BASE64)
        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_high_resolution_small_jpeg_is_resized(self):
        # 低字节但高分辨率 JPEG 仍需按长边缩放，避免 OCR 超时/失败。
        large_jpeg = _jpeg_base64(2200, 1600)
        data_uri = ocr_engine._prepare_image_data_uri(large_jpeg, max_edge=1280)
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as img:
            self.assertLessEqual(max(img.size), 1280)

    def test_english_name_with_co_substring_is_kept(self):
        for name in (
            "Nicole Wang",
            "Lincoln Park",
            "Marco Rossi",
            "Addison Wang",
            "Broadway Chen",
            "Ismail Hassan",
            "Maile Chen",
            "Alex Groom",
        ):
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(
                    [
                        name,
                        "Sales Manager",
                        "13800138000",
                        "person@example.com",
                    ]
                )
                self.assertEqual(name, result["姓名"])

    def test_two_char_cn_name_with_lu_is_kept_and_not_used_as_address(self):
        for name in ("张路", "马路"):
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(
                    [
                        "上海玖协机械有限公司",
                        name,
                        "销售经理",
                        "13800138000",
                        "中山路128号",
                    ]
                )
                self.assertEqual(name, result["姓名"])
                self.assertEqual("中山路128号", result["地址"])

    def test_company_line_with_city_or_road_does_not_steal_address_or_title(self):
        cases = (
            (
                [
                    "深圳市中山科技有限公司",
                    "李明",
                    "销售经理",
                    "13800138000",
                    "中山路128号",
                ],
                "李明",
                "深圳市中山科技有限公司",
                "销售经理",
                "中山路128号",
            ),
            (
                [
                    "广州市天河区创新有限公司",
                    "王芳",
                    "总监",
                    "天河路88号",
                ],
                "王芳",
                "广州市天河区创新有限公司",
                "总监",
                "天河路88号",
            ),
            (
                [
                    "浦东新区销售有限公司",
                    "钱伟",
                    "顾问",
                    "世纪大道1号",
                ],
                "钱伟",
                "浦东新区销售有限公司",
                "顾问",
                "世纪大道1号",
            ),
            (
                [
                    "上海销售有限公司",
                    "周杰",
                    "工程师",
                    "浦东大道1号",
                ],
                "周杰",
                "上海销售有限公司",
                "工程师",
                "浦东大道1号",
            ),
        )
        for chunks, name, company, title, address in cases:
            with self.subTest(company=company):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual(company, result["公司"])
                self.assertEqual(title, result["职位"])
                self.assertEqual(address, result["地址"])

    def test_mixed_company_address_line_is_kept_when_no_dedicated_address(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海玖协机械有限公司 嘉定区宝安公路4999号",
                "赵美娜",
                "销售经理",
            ]
        )
        self.assertEqual("赵美娜", result["姓名"])
        self.assertEqual("上海玖协机械有限公司 嘉定区宝安公路4999号", result["公司"])
        self.assertEqual("销售经理", result["职位"])
        self.assertEqual("上海玖协机械有限公司 嘉定区宝安公路4999号", result["地址"])

    def test_title_or_contact_line_does_not_steal_street_address(self):
        cases = (
            (
                [
                    "上海玖协机械有限公司",
                    "李明",
                    "区域经理",
                    "13800138000",
                    "中山路128号",
                ],
                "李明",
                "区域经理",
                "中山路128号",
            ),
            (
                [
                    "上海玖协机械有限公司",
                    "王芳",
                    "市场总监",
                    "天河路88号",
                ],
                "王芳",
                "市场总监",
                "天河路88号",
            ),
            (
                [
                    "上海建设集团有限公司",
                    "赵强",
                    "市政工程师",
                    "解放路1号",
                ],
                "赵强",
                "市政工程师",
                "解放路1号",
            ),
            (
                [
                    "上海玖协机械有限公司",
                    "张三",
                    "销售经理",
                    "手机号：13800138000",
                    "中山路128号",
                ],
                "张三",
                "销售经理",
                "中山路128号",
            ),
            (
                [
                    "上海玖协机械有限公司",
                    "张三",
                    "工程师",
                    "工号：10086",
                    "中山路128号",
                ],
                "张三",
                "工程师",
                "中山路128号",
            ),
            (
                [
                    "上海玖协机械有限公司",
                    "张三",
                    "销售经理",
                    "电话号码：021-12345678",
                    "中山路128号",
                ],
                "张三",
                "销售经理",
                "中山路128号",
            ),
            (
                [
                    "上海玖协机械有限公司",
                    "李明",
                    "经理",
                    "微信号：wxid_abc",
                    "中山路128号",
                ],
                "李明",
                "经理",
                "中山路128号",
            ),
        )
        for chunks, name, title, address in cases:
            with self.subTest(title=title, decoy=chunks[-2]):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual(title, result["职位"])
                self.assertEqual(address, result["地址"])

    def test_street_line_outranks_district_only_line(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "钱伟",
                "顾问",
                "浦东新区",
                "世纪大道1号",
            ]
        )
        self.assertEqual("钱伟", result["姓名"])
        self.assertEqual("顾问", result["职位"])
        self.assertEqual("世纪大道1号", result["地址"])

    def test_mixed_address_and_phone_line_is_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海玖协机械有限公司",
                "李明",
                "经理",
                "地址：中山路128号 电话：021-12345678",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("地址：中山路128号 电话：021-12345678", result["地址"])

    def test_street_line_still_is_not_treated_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "中山路128号",
                "销售经理",
                "13800138000",
            ]
        )
        self.assertEqual("", result["姓名"])
        self.assertEqual("中山路128号", result["地址"])

    def test_address_label_still_is_not_treated_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Add: 88 West Road",
                "Sales Manager",
                "13800138000",
            ]
        )
        self.assertEqual("", result["姓名"])
        self.assertEqual("Add: 88 West Road", result["地址"])

    def test_english_road_and_director_are_mapped(self):
        result = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "Director",
                "Acme Corporation",
                "88 Queen's Road Central",
                "john@acme.com",
            ]
        )
        self.assertEqual("John Smith", result["姓名"])
        self.assertEqual("Acme Corporation", result["公司"])
        self.assertEqual("Director", result["职位"])
        self.assertEqual("88 Queen's Road Central", result["地址"])
        self.assertEqual("john@acme.com", result["邮箱"])

    def test_english_street_line_is_address_not_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Manager",
                "Acme Corporation",
                "123 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", result["姓名"])
        self.assertEqual("Manager", result["职位"])
        self.assertEqual("123 Main Street", result["地址"])

    def test_broadway_name_is_not_treated_as_address(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Broadway Chen",
                "Sales Manager",
                "13800138000",
            ]
        )
        self.assertEqual("Broadway Chen", result["姓名"])
        self.assertEqual("", result["地址"])

    def test_building_line_is_preferred_over_district_only(self):
        result = ocr_engine._map_business_card_fields(
            [
                "某某科技有限公司",
                "李明",
                "产品经理",
                "浦东新区",
                "科技大厦A座18楼",
                "13800138000",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("产品经理", result["职位"])
        self.assertEqual("科技大厦A座18楼", result["地址"])

    def test_avenue_line_is_address_and_not_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某科技有限公司",
                "滨江大道1888",
                "李明",
                "产品经理",
                "13800138000",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("产品经理", result["职位"])
        self.assertEqual("滨江大道1888", result["地址"])

    def test_avenue_line_outranks_district_only_line(self):
        result = ocr_engine._map_business_card_fields(
            [
                "某某科技有限公司",
                "李明",
                "产品经理",
                "浦东新区",
                "滨江大道1888",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("产品经理", result["职位"])
        self.assertEqual("滨江大道1888", result["地址"])

    def test_building_block_line_outranks_district_only(self):
        result = ocr_engine._map_business_card_fields(
            [
                "某某科技有限公司",
                "李明",
                "产品经理",
                "浦东新区",
                "A座18楼",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("产品经理", result["职位"])
        self.assertEqual("A座18楼", result["地址"])

    def test_building_dong_line_is_address(self):
        result = ocr_engine._map_business_card_fields(
            [
                "某某科技有限公司",
                "李明",
                "经理",
                "3栋2单元501",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("3栋2单元501", result["地址"])

    def test_two_char_name_with_dao_is_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "王道",
                "经理",
                "中山路128号",
            ]
        )
        self.assertEqual("王道", result["姓名"])
        self.assertEqual("经理", result["职位"])
        self.assertEqual("中山路128号", result["地址"])

    def test_building_manager_title_does_not_steal_street(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海玖协机械有限公司",
                "李明",
                "楼宇经理",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("楼宇经理", result["职位"])
        self.assertEqual("中山路128号", result["地址"])

    def test_email_label_still_is_not_treated_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Email: person@example.com",
                "Sales Manager",
                "13800138000",
            ]
        )
        self.assertNotIn("@", result["姓名"])
        self.assertNotEqual("person@example.com", result["姓名"])

    def test_extract_text_chunks_reads_nested_string_layout(self):
        chunks = ocr_engine._extract_text_chunks(
            {
                "data": {
                    "md_results": "发票号码 123456\n价税合计金额 100.00",
                    "layout_details": [[{"content": "购买方名称 测试公司"}]],
                }
            }
        )
        self.assertEqual(
            chunks,
            [
                "发票号码 123456",
                "价税合计金额 100.00",
                "购买方名称 测试公司",
            ],
        )

    @patch("ocr_engine.requests.post")
    def test_invoice_layout_text_is_structured_with_expected_fields(self, post):
        fields = TEMPLATE_MAP["invoice"]["fields"]
        post.side_effect = [
            mock_response(
                {
                    "data": {
                        "md_results": (
                            "发票号码：3100261130\n"
                            "开票日期：2026年6月19日\n"
                            "购买方名称：上海测试科技有限公司\n"
                            "销售方名称：北京样例服务有限公司\n"
                            "价税合计：¥123.45\n"
                            "税额：11.23"
                        ),
                        "layout_details": [],
                    }
                }
            ),
            mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "发票号码": "3100261130",
                                        "开票日期": "2026-06-19",
                                        "购买方名称": "上海测试科技有限公司",
                                        "销售方名称": "北京样例服务有限公司",
                                        "价税合计金额": "123.45",
                                        "税额": "11.23",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            ),
        ]

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            TEMPLATE_MAP["invoice"]["template"],
            expected_fields=fields,
            template_type="invoice",
        )

        self.assertEqual("3100261130", result["发票号码"])
        self.assertEqual("123.45", result["价税合计金额"])
        self.assertEqual(2, post.call_count)
        layout_payload = post.call_args_list[0].kwargs["json"]
        self.assertTrue(layout_payload["file"].startswith("data:image/png;base64,"))
        structure_payload = post.call_args_list[1].kwargs["json"]
        self.assertIn("OCR文本如下", structure_payload["messages"][1]["content"])

    @patch("ocr_engine.requests.post")
    def test_business_card_uses_nested_md_results_string_without_chat_call(self, post):
        post.return_value = mock_response(
            {
                "data": {
                    "md_results": "Nicole Wang\nManager\n13800138000\nnicole@example.com",
                    "layout_details": [],
                }
            }
        )

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            TEMPLATE_MAP["business_card"]["template"],
            expected_fields=TEMPLATE_MAP["business_card"]["fields"],
            template_type="business_card",
        )

        self.assertEqual("Nicole Wang", result["姓名"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("nicole@example.com", result["邮箱"])
        self.assertEqual(1, post.call_count)

    @patch("ocr_engine.requests.post")
    def test_custom_field_cannot_collide_with_layout_metadata(self, post):
        post.side_effect = [
            mock_response(
                {
                    "id": "layout-request-id",
                    "model": "glm-ocr",
                    "md_results": "产品模型 GLM-5",
                }
            ),
            mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"model":"GLM-5"}',
                            }
                        }
                    ]
                }
            ),
        ]

        result = ocr_engine.call_llm(
            SMALL_PNG_BASE64,
            "抽取 model",
            expected_fields=["model"],
            template_type="custom",
        )

        self.assertEqual(result, {"model": "GLM-5"})
        self.assertEqual(post.call_count, 2)

    def test_mobile_and_landline_with_separators_are_kept(self):
        cases = (
            (["上海某某科技有限公司", "李明", "经理", "138-0013-8000", "中山路128号"], "13800138000", ""),
            (["上海某某科技有限公司", "李明", "经理", "138 0013 8000", "中山路128号"], "13800138000", ""),
            (["上海某某科技有限公司", "李明", "经理", "+86-138-0013-8000", "中山路128号"], "13800138000", ""),
            (["上海某某科技有限公司", "李明", "经理", "021-1234-5678", "中山路128号"], "", "021-1234-5678"),
            (["上海某某科技有限公司", "李明", "经理", "021 1234 5678", "中山路128号"], "", "021-1234-5678"),
        )
        for chunks, mobile, landline in cases:
            with self.subTest(phone=chunks[3]):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual("李明", result["姓名"])
                self.assertEqual(mobile, result["手机"])
                self.assertEqual(landline, result["座机"])
                self.assertEqual("中山路128号", result["地址"])

    def test_english_lane_drive_suite_are_address_not_name(self):
        cases = (
            (
                ["John Smith", "Director", "Acme Corporation", "88 Oak Lane"],
                "John Smith",
                "Director",
                "88 Oak Lane",
            ),
            (
                ["Jane Doe", "Manager", "Acme Corporation", "120 Park Drive"],
                "Jane Doe",
                "Manager",
                "120 Park Drive",
            ),
            (
                ["Jane Doe", "Manager", "Acme Corporation", "Suite 1201"],
                "Jane Doe",
                "Manager",
                "Suite 1201",
            ),
            (
                ["John Smith", "Director", "Acme Corporation", "88 Oak Way"],
                "John Smith",
                "Director",
                "88 Oak Way",
            ),
        )
        for chunks, name, title, address in cases:
            with self.subTest(address=address):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual("Acme Corporation", result["公司"])
                self.assertEqual(title, result["职位"])
                self.assertEqual(address, result["地址"])

    def test_lane_as_given_name_is_not_treated_as_address(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Lane Cooper",
                "Manager",
                "Acme Corporation",
                "88 West Road",
            ]
        )
        self.assertEqual("Lane Cooper", result["姓名"])
        self.assertEqual("Manager", result["职位"])
        self.assertEqual("88 West Road", result["地址"])

    def test_floor_room_and_alley_outrank_district_only_line(self):
        cases = (
            (
                ["某某科技有限公司", "李明", "产品经理", "浦东新区", "18层1201室"],
                "18层1201室",
            ),
            (
                ["某某科技有限公司", "李明", "经理", "浦东新区", "翠湖巷"],
                "翠湖巷",
            ),
        )
        for chunks, address in cases:
            with self.subTest(address=address):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual("李明", result["姓名"])
                self.assertEqual(address, result["地址"])

    def test_all_caps_english_name_and_ceo_title_are_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "JOHN SMITH",
                "CEO",
                "ACME CORPORATION",
                "88 Queen's Road Central",
                "JOHN@ACME.COM",
            ]
        )
        self.assertEqual("JOHN SMITH", result["姓名"])
        self.assertEqual("ACME CORPORATION", result["公司"])
        self.assertEqual("CEO", result["职位"])
        self.assertEqual("88 Queen's Road Central", result["地址"])
        self.assertEqual("JOHN@ACME.COM", result["邮箱"])

    def test_chairman_title_is_mapped(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "李明",
                "董事长",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("董事长", result["职位"])
        self.assertEqual("中山路128号", result["地址"])

    def test_parenthesized_and_dotted_landlines_are_kept(self):
        cases = (
            (["上海某某科技有限公司", "李明", "经理", "(021)12345678", "中山路128号"], "021-12345678"),
            (["上海某某科技有限公司", "李明", "经理", "(021) 1234-5678", "中山路128号"], "021-1234-5678"),
            (["上海某某科技有限公司", "李明", "经理", "（010）88886666", "中山路128号"], "010-88886666"),
            (["上海某某科技有限公司", "李明", "经理", "021.12345678", "中山路128号"], "021-12345678"),
        )
        for chunks, landline in cases:
            with self.subTest(phone=chunks[3]):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual("李明", result["姓名"])
                self.assertEqual(landline, result["座机"])
                self.assertEqual("中山路128号", result["地址"])

    def test_fax_line_does_not_steal_landline(self):
        cases = (
            (
                [
                    "上海某某科技有限公司",
                    "李明",
                    "经理",
                    "传真：021-12345678",
                    "电话：021-87654321",
                    "中山路128号",
                ],
                "021-87654321",
            ),
            (
                [
                    "Acme Corporation",
                    "John Smith",
                    "Director",
                    "Fax: 021-12345678",
                    "Tel: 021-87654321",
                    "88 West Road",
                ],
                "021-87654321",
            ),
        )
        for chunks, landline in cases:
            with self.subTest(decoy=chunks[3]):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(landline, result["座机"])

    def test_fax_is_kept_when_it_is_the_only_landline(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某科技有限公司",
                "李明",
                "经理",
                "传真：021-12345678",
                "中山路128号",
            ]
        )
        self.assertEqual("021-12345678", result["座机"])

    def test_dotted_mobile_is_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某科技有限公司",
                "李明",
                "经理",
                "138.0013.8000",
                "中山路128号",
            ]
        )
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("中山路128号", result["地址"])

    def test_english_rd_st_dr_plaza_square_are_address_not_name(self):
        cases = (
            (
                ["John Smith", "Director", "Acme Corporation", "88 West Rd."],
                "John Smith",
                "Director",
                "88 West Rd.",
            ),
            (
                ["Jane Doe", "Manager", "Acme Corporation", "120 Park St"],
                "Jane Doe",
                "Manager",
                "120 Park St",
            ),
            (
                ["Jane Doe", "Manager", "Acme Corporation", "15 Oak Dr"],
                "Jane Doe",
                "Manager",
                "15 Oak Dr",
            ),
            (
                ["John Smith", "Director", "Acme Corporation", "1 Times Square"],
                "John Smith",
                "Director",
                "1 Times Square",
            ),
            (
                ["Jane Doe", "Manager", "Acme Corporation", "200 Park Plaza"],
                "Jane Doe",
                "Manager",
                "200 Park Plaza",
            ),
            (
                ["John Smith", "Director", "Acme Corporation", "500 Oak Parkway"],
                "John Smith",
                "Director",
                "500 Oak Parkway",
            ),
            (
                ["John Smith", "Director", "Acme Corporation", "10 Columbus Circle"],
                "John Smith",
                "Director",
                "10 Columbus Circle",
            ),
        )
        for chunks, name, title, address in cases:
            with self.subTest(address=address):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual("Acme Corporation", result["公司"])
                self.assertEqual(title, result["职位"])
                self.assertEqual(address, result["地址"])

    def test_st_and_dr_as_name_prefix_are_not_treated_as_address(self):
        cases = (
            (["St John", "Manager", "Acme Corporation", "88 West Road"], "St John"),
            (["Dr John Smith", "Director", "Acme Corporation", "88 West Road"], "Dr John Smith"),
        )
        for chunks, name in cases:
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual("88 West Road", result["地址"])

    def test_science_park_and_plaza_outrank_district_only_line(self):
        cases = (
            (
                ["上海某某科技有限公司", "李明", "产品经理", "张江科技园"],
                "张江科技园",
            ),
            (
                ["上海某某科技有限公司", "李明", "经理", "人民广场"],
                "人民广场",
            ),
            (
                ["上海某某科技有限公司", "李明", "产品经理", "浦东新区", "张江科技园"],
                "张江科技园",
            ),
            (
                ["上海某某科技有限公司", "李明", "经理", "黄浦区", "人民广场"],
                "人民广场",
            ),
        )
        for chunks, address in cases:
            with self.subTest(address=address):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual("李明", result["姓名"])
                self.assertEqual(address, result["地址"])

    def test_two_char_name_with_yuan_is_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某科技有限公司",
                "李园",
                "经理",
                "中山路128号",
            ]
        )
        self.assertEqual("李园", result["姓名"])
        self.assertEqual("经理", result["职位"])
        self.assertEqual("中山路128号", result["地址"])

    def test_founder_and_designer_titles_are_mapped(self):
        en = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "Founder",
                "Acme Corporation",
                "88 West Road",
            ]
        )
        self.assertEqual("John Smith", en["姓名"])
        self.assertEqual("Founder", en["职位"])
        self.assertEqual("88 West Road", en["地址"])

        cn = ocr_engine._map_business_card_fields(
            [
                "上海某某科技有限公司",
                "李明",
                "产品设计师",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", cn["姓名"])
        self.assertEqual("产品设计师", cn["职位"])
        self.assertEqual("中山路128号", cn["地址"])

        architect = ocr_engine._map_business_card_fields(
            [
                "上海某某科技有限公司",
                "李明",
                "架构师",
                "中山路128号",
            ]
        )
        self.assertEqual("架构师", architect["职位"])

    def test_middle_initial_english_name_is_kept(self):
        for name in ("John A. Smith", "John A Smith"):
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(
                    [
                        name,
                        "Director",
                        "Acme Corporation",
                        "88 Queen's Road",
                    ]
                )
                self.assertEqual(name, result["姓名"])
                self.assertEqual("Director", result["职位"])
                self.assertEqual("Acme Corporation", result["公司"])
                self.assertNotIn("\n", result["姓名"])
                self.assertNotIn("Director", result["姓名"])
                self.assertNotIn("Acme", result["姓名"])

    def test_hyphenated_english_given_name_is_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "Mary-Jane Watson",
                "Manager",
                "Acme Inc",
                "100 Park Ave",
            ]
        )
        self.assertEqual("Mary-Jane Watson", result["姓名"])
        self.assertEqual("Manager", result["职位"])

    def test_llc_limited_llp_are_companies_not_names(self):
        cases = (
            (
                [
                    "Northwind Analytics LLC",
                    "Jane Smith",
                    "Partner",
                    "100 Park Ave",
                    "jane@northwind.com",
                ],
                "Jane Smith",
                "Northwind Analytics LLC",
                "Partner",
                "100 Park Ave",
            ),
            (
                [
                    "Acme Trading Limited",
                    "Jane Smith",
                    "Manager",
                    "100 Park Ave",
                ],
                "Jane Smith",
                "Acme Trading Limited",
                "Manager",
                "100 Park Ave",
            ),
            (
                [
                    "Northwind LLP",
                    "Jane Smith",
                    "Partner",
                    "100 Park Ave",
                ],
                "Jane Smith",
                "Northwind LLP",
                "Partner",
                "100 Park Ave",
            ),
            (
                [
                    "Siemens GmbH",
                    "Hans Mueller",
                    "Manager",
                    "100 Park Ave",
                ],
                "Hans Mueller",
                "Siemens GmbH",
                "Manager",
                "100 Park Ave",
            ),
        )
        for chunks, name, company, title, address in cases:
            with self.subTest(company=company):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual(company, result["公司"])
                self.assertEqual(title, result["职位"])
                self.assertEqual(address, result["地址"])

    def test_law_firm_and_institute_are_companies_not_names(self):
        law = ocr_engine._map_business_card_fields(
            [
                "某某律师事务所",
                "李明",
                "律师",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", law["姓名"])
        self.assertEqual("某某律师事务所", law["公司"])
        self.assertEqual("律师", law["职位"])
        self.assertEqual("中山路128号", law["地址"])

        studio = ocr_engine._map_business_card_fields(
            [
                "某某设计工作室",
                "李明",
                "设计师",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", studio["姓名"])
        self.assertEqual("某某设计工作室", studio["公司"])
        self.assertEqual("设计师", studio["职位"])

        institute = ocr_engine._map_business_card_fields(
            [
                "中科院上海研究所",
                "李明",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("李明", institute["姓名"])
        self.assertEqual("中科院上海研究所", institute["公司"])
        self.assertNotEqual("中科院上", institute["姓名"])

        university = ocr_engine._map_business_card_fields(
            [
                "复旦大学",
                "李明",
                "经理",
            ]
        )
        self.assertEqual("李明", university["姓名"])
        self.assertEqual("复旦大学", university["公司"])
        self.assertNotEqual("复旦大学", university["姓名"])

    def test_partner_and_lawyer_titles_are_mapped(self):
        cn = ocr_engine._map_business_card_fields(
            [
                "某某律师事务所",
                "李明",
                "合伙人",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", cn["姓名"])
        self.assertEqual("合伙人", cn["职位"])

        en = ocr_engine._map_business_card_fields(
            [
                "Jane Smith",
                "Partner",
                "Northwind LLP",
                "100 Park Ave",
            ]
        )
        self.assertEqual("Jane Smith", en["姓名"])
        self.assertEqual("Partner", en["职位"])
        self.assertEqual("Northwind LLP", en["公司"])

    def test_fullwidth_and_spaced_emails_are_kept(self):
        cases = (
            (["李明", "经理", "上海某某有限公司", "zhangsan＠example.com", "中山路128号"], "zhangsan@example.com"),
            (["李明", "经理", "上海某某有限公司", "zhangsan @ example.com", "中山路128号"], "zhangsan@example.com"),
            (["李明", "经理", "li.ming＠example．com"], "li.ming@example.com"),
        )
        for chunks, email in cases:
            with self.subTest(raw=chunks[-1] if "@" in chunks[-1] or "＠" in chunks[-1] else chunks[-2]):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual("李明", result["姓名"])
                self.assertEqual(email, result["邮箱"])

    def test_service_hotline_is_mapped_to_landline(self):
        cases = (
            (["上海某某有限公司", "李明", "经理", "400-800-1234", "中山路128号"], "", "400-800-1234"),
            (["上海某某有限公司", "李明", "经理", "400 888 1234", "中山路128号"], "", "400-888-1234"),
            (["上海某某有限公司", "李明", "经理", "800-810-8888"], "", "800-810-8888"),
            (["上海某某有限公司", "李明", "经理", "400-800-1234", "13800138000"], "13800138000", "400-800-1234"),
        )
        for chunks, mobile, landline in cases:
            with self.subTest(phone=chunks[3]):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual("李明", result["姓名"])
                self.assertEqual(mobile, result["手机"])
                self.assertEqual(landline, result["座机"])

    def test_slash_separated_mobile_is_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "李明",
                "经理",
                "138/0013/8000",
                "中山路128号",
            ]
        )
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("李明", result["姓名"])

    def test_international_prefix_mobile_does_not_fill_landline(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "李明",
                "经理",
                "0086-138-0013-8000",
                "中山路128号",
            ]
        )
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("", result["座机"])

    def test_center_and_tower_outrank_district_only_line(self):
        cases = (
            (
                ["某某科技有限公司", "李明", "产品经理", "浦东新区", "陆家嘴金融中心"],
                "陆家嘴金融中心",
            ),
            (
                ["某某科技有限公司", "李明", "经理", "国贸商务中心"],
                "国贸商务中心",
            ),
            (
                ["Jane Smith", "Manager", "Acme Inc", "One World Trade Center"],
                "One World Trade Center",
            ),
            (
                ["Jane Smith", "Manager", "Acme Inc", "IFC Tower"],
                "IFC Tower",
            ),
            (
                ["Jane Smith", "Manager", "Acme Inc", "浦东新区", "200 Park Center"],
                "200 Park Center",
            ),
        )
        for chunks, address in cases:
            with self.subTest(address=address):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(address, result["地址"])
                self.assertNotEqual("浦东新区", result["地址"])


    def test_hospital_bank_school_are_companies_not_names(self):
        hospital = ocr_engine._map_business_card_fields(
            [
                "瑞金医院",
                "李明",
                "主任医师",
                "瑞金二路197号",
            ]
        )
        self.assertEqual("李明", hospital["姓名"])
        self.assertEqual("瑞金医院", hospital["公司"])
        self.assertEqual("主任医师", hospital["职位"])
        self.assertEqual("瑞金二路197号", hospital["地址"])
        self.assertNotEqual("瑞金医院", hospital["姓名"])

        bank = ocr_engine._map_business_card_fields(
            [
                "中国工商银行",
                "张伟",
                "客户经理",
                "南京东路100号",
            ]
        )
        self.assertEqual("张伟", bank["姓名"])
        self.assertEqual("中国工商银行", bank["公司"])
        self.assertEqual("客户经理", bank["职位"])
        self.assertEqual("南京东路100号", bank["地址"])

        school = ocr_engine._map_business_card_fields(
            [
                "北京市第一中学",
                "王芳",
                "校长",
                "西城区长椿街7号",
            ]
        )
        self.assertEqual("王芳", school["姓名"])
        self.assertEqual("北京市第一中学", school["公司"])
        self.assertEqual("校长", school["职位"])
        self.assertEqual("西城区长椿街7号", school["地址"])
        self.assertNotEqual("北京市第一中学", school["地址"])

        college = ocr_engine._map_business_card_fields(
            [
                "上海财经学院",
                "赵强",
                "教授",
                "国定路777号",
            ]
        )
        self.assertEqual("赵强", college["姓名"])
        self.assertEqual("上海财经学院", college["公司"])
        self.assertEqual("教授", college["职位"])
        self.assertEqual("国定路777号", college["地址"])

    def test_association_and_foundation_are_companies_not_titles(self):
        association = ocr_engine._map_business_card_fields(
            [
                "上海律师协会",
                "李明",
                "秘书长",
                "中山路1号",
            ]
        )
        self.assertEqual("李明", association["姓名"])
        self.assertEqual("上海律师协会", association["公司"])
        self.assertEqual("秘书长", association["职位"])
        self.assertNotEqual("上海律师协会", association["职位"])
        self.assertEqual("中山路1号", association["地址"])

        foundation = ocr_engine._map_business_card_fields(
            [
                "某某基金会",
                "李明",
                "理事长",
                "中山路1号",
            ]
        )
        self.assertEqual("李明", foundation["姓名"])
        self.assertEqual("某某基金会", foundation["公司"])
        self.assertNotEqual("某某基金", foundation["姓名"])
        self.assertEqual("中山路1号", foundation["地址"])

        branch = ocr_engine._map_business_card_fields(
            [
                "招商银行上海分行",
                "李明",
                "行长",
                "中山路1号",
            ]
        )
        self.assertEqual("李明", branch["姓名"])
        self.assertEqual("招商银行上海分行", branch["公司"])
        self.assertEqual("行长", branch["职位"])
        self.assertNotEqual("招商银行", branch["姓名"])

    def test_english_hospital_university_are_companies_not_names(self):
        university = ocr_engine._map_business_card_fields(
            [
                "Harvard University",
                "Jane Doe",
                "Professor",
                "1350 Massachusetts Avenue",
            ]
        )
        self.assertEqual("Jane Doe", university["姓名"])
        self.assertEqual("Harvard University", university["公司"])
        self.assertEqual("Professor", university["职位"])
        self.assertEqual("1350 Massachusetts Avenue", university["地址"])
        self.assertNotEqual("Harvard University", university["姓名"])

        hospital = ocr_engine._map_business_card_fields(
            [
                "Massachusetts General Hospital",
                "Jane Doe",
                "Director",
                "55 Fruit Street",
            ]
        )
        self.assertEqual("Jane Doe", hospital["姓名"])
        self.assertEqual("Massachusetts General Hospital", hospital["公司"])
        self.assertEqual("55 Fruit Street", hospital["地址"])

        institute = ocr_engine._map_business_card_fields(
            [
                "Lincoln Institute",
                "Jane Doe",
                "Analyst",
                "244 Wood Street",
            ]
        )
        self.assertEqual("Jane Doe", institute["姓名"])
        self.assertEqual("Lincoln Institute", institute["公司"])
        self.assertEqual("Analyst", institute["职位"])

    def test_corp_group_holdings_are_companies(self):
        cases = (
            (
                ["John Smith", "CEO", "Acme Corp", "100 Main Street"],
                "John Smith",
                "Acme Corp",
            ),
            (
                ["John Smith", "CEO", "Alibaba Group", "100 Main Street"],
                "John Smith",
                "Alibaba Group",
            ),
            (
                ["John Smith", "CEO", "SoftBank Holdings", "100 Main Street"],
                "John Smith",
                "SoftBank Holdings",
            ),
            (
                ["John Smith", "CEO", "Acme PLC", "100 Main Street"],
                "John Smith",
                "Acme PLC",
            ),
        )
        for chunks, name, company in cases:
            with self.subTest(company=company):
                result = ocr_engine._map_business_card_fields(chunks)
                self.assertEqual(name, result["姓名"])
                self.assertEqual(company, result["公司"])
                self.assertEqual("CEO", result["职位"])
                self.assertEqual("100 Main Street", result["地址"])

    def test_engineer_consultant_professor_titles_are_mapped(self):
        engineer = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "Software Engineer",
                "Acme Inc",
                "100 Park Ave",
            ]
        )
        self.assertEqual("John Smith", engineer["姓名"])
        self.assertEqual("Software Engineer", engineer["职位"])
        self.assertEqual("Acme Inc", engineer["公司"])

        consultant = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "Consultant",
                "Acme Inc",
                "100 Park Ave",
            ]
        )
        self.assertEqual("Consultant", consultant["职位"])

        professor = ocr_engine._map_business_card_fields(
            [
                "复旦大学",
                "李明",
                "教授",
                "邯郸路220号",
            ]
        )
        self.assertEqual("李明", professor["姓名"])
        self.assertEqual("复旦大学", professor["公司"])
        self.assertEqual("教授", professor["职位"])

        chairman = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "Chairman",
                "Acme Inc",
                "100 Park Ave",
            ]
        )
        self.assertEqual("Chairman", chairman["职位"])

        cmo = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "CMO",
                "Acme Inc",
                "100 Park Ave",
            ]
        )
        self.assertEqual("CMO", cmo["职位"])

    def test_apostrophe_english_name_is_kept(self):
        for name in ("Patrick O'Brien", "Mary O'Connor"):
            with self.subTest(name=name):
                result = ocr_engine._map_business_card_fields(
                    [
                        name,
                        "Manager",
                        "Acme Inc",
                        "100 Park Ave",
                    ]
                )
                self.assertEqual(name, result["姓名"])
                self.assertEqual("Manager", result["职位"])
                self.assertEqual("Acme Inc", result["公司"])
                self.assertNotIn(result["姓名"], ("Patrick O", "Mary O"))

        curly = ocr_engine._map_business_card_fields(
            [
                "Patrick O\u2019Brien",
                "Manager",
                "Acme Inc",
                "100 Park Ave",
            ]
        )
        self.assertEqual("Patrick O'Brien", curly["姓名"])

    def test_po_box_and_room_abbreviation_are_address(self):
        pobox = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Manager",
                "Acme Inc",
                "P.O. Box 1234",
            ]
        )
        self.assertEqual("Jane Doe", pobox["姓名"])
        self.assertEqual("P.O. Box 1234", pobox["地址"])

        pobox2 = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Manager",
                "Acme Inc",
                "PO Box 88",
            ]
        )
        self.assertEqual("PO Box 88", pobox2["地址"])

        room = ocr_engine._map_business_card_fields(
            [
                "John Smith",
                "CEO",
                "Acme Corporation",
                "Rm 1201",
            ]
        )
        self.assertEqual("John Smith", room["姓名"])
        self.assertEqual("Rm 1201", room["地址"])

    def test_fullwidth_mobile_digits_are_kept(self):
        result = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "李明",
                "经理",
                "１３８００１３８０００",
                "中山路128号",
            ]
        )
        self.assertEqual("李明", result["姓名"])
        self.assertEqual("13800138000", result["手机"])
        self.assertEqual("中山路128号", result["地址"])


class HandlerTest(unittest.TestCase):
    def test_handler_rejects_oversized_base64_before_calling_llm(self):
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": "A" * (MAX_IMAGE_SIZE + 1),
                        "template_type": "business_card",
                    }
                )
            },
            None,
        )

        self.assertEqual(413, response["statusCode"])
        self.assertFalse(json.loads(response["body"])["success"])

    @patch("index.call_llm")
    def test_handler_passes_template_context_and_rejects_empty_result(self, call_llm):
        call_llm.return_value = {}
        response = index.handler(
            {
                "body": json.dumps(
                    {
                        "image_base64": SMALL_PNG_BASE64,
                        "template_type": "invoice",
                    }
                )
            },
            None,
        )

        body = json.loads(response["body"])
        self.assertEqual(422, response["statusCode"])
        self.assertFalse(body["success"])
        self.assertIn("未识别到有效字段", body["error"])
        _, kwargs = call_llm.call_args
        self.assertEqual(kwargs["template_type"], "invoice")
        self.assertEqual(
            kwargs["expected_fields"],
            TEMPLATE_MAP["invoice"]["fields"],
        )


if __name__ == "__main__":
    unittest.main()
