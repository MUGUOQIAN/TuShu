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

    def test_tech_shares_clinic_and_kindergarten_are_companies_not_names(self):
        tech = ocr_engine._map_business_card_fields(
            [
                "杭州云启科技",
                "张三",
                "产品经理",
                "13800138000",
                "文一西路969号",
            ]
        )
        self.assertEqual("张三", tech["姓名"])
        self.assertEqual("杭州云启科技", tech["公司"])
        self.assertEqual("产品经理", tech["职位"])
        self.assertEqual("文一西路969号", tech["地址"])
        self.assertNotEqual("杭州云启", tech["姓名"])

        shares = ocr_engine._map_business_card_fields(
            [
                "比亚迪股份",
                "王五",
                "经理",
                "13800138000",
                "坪山路1号",
            ]
        )
        self.assertEqual("王五", shares["姓名"])
        self.assertEqual("比亚迪股份", shares["公司"])
        self.assertNotEqual("比亚迪股", shares["姓名"])

        holdings = ocr_engine._map_business_card_fields(
            [
                "腾讯控股",
                "李四",
                "总监",
                "13800138000",
                "高新路1号",
            ]
        )
        self.assertEqual("李四", holdings["姓名"])
        self.assertEqual("腾讯控股", holdings["公司"])
        self.assertNotEqual("腾讯控股", holdings["姓名"])

        research = ocr_engine._map_business_card_fields(
            [
                "微软亚洲研究院",
                "周杰",
                "研究员",
                "13800138000",
                "丹棱街5号",
            ]
        )
        self.assertEqual("周杰", research["姓名"])
        self.assertEqual("微软亚洲研究院", research["公司"])
        self.assertEqual("研究员", research["职位"])
        self.assertNotEqual("微软亚洲", research["姓名"])

        clinic = ocr_engine._map_business_card_fields(
            [
                "阳光口腔诊所",
                "赵六",
                "医师",
                "13800138000",
                "中山路1号",
            ]
        )
        self.assertEqual("赵六", clinic["姓名"])
        self.assertEqual("阳光口腔诊所", clinic["公司"])
        self.assertEqual("医师", clinic["职位"])
        self.assertNotEqual("阳光口腔", clinic["姓名"])

        kindergarten = ocr_engine._map_business_card_fields(
            [
                "市第一幼儿园",
                "钱伟",
                "园长",
                "13800138000",
                "中山路1号",
            ]
        )
        self.assertEqual("钱伟", kindergarten["姓名"])
        self.assertEqual("市第一幼儿园", kindergarten["公司"])
        self.assertEqual("园长", kindergarten["职位"])
        self.assertEqual("中山路1号", kindergarten["地址"])
        self.assertNotEqual("市第一幼儿园", kindergarten["地址"])

        tech_title = ocr_engine._map_business_card_fields(
            [
                "杭州云启科技",
                "张三",
                "科技经理",
                "13800138000",
                "文一西路969号",
            ]
        )
        self.assertEqual("张三", tech_title["姓名"])
        self.assertEqual("杭州云启科技", tech_title["公司"])
        self.assertEqual("科技经理", tech_title["职位"])

    def test_title_and_name_on_same_line_are_split(self):
        cn_title_first = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "销售总监 张三",
                "13800138000",
                "中山路128号",
            ]
        )
        self.assertEqual("张三", cn_title_first["姓名"])
        self.assertEqual("销售总监 张三", cn_title_first["职位"])
        self.assertEqual("上海某某有限公司", cn_title_first["公司"])

        cn_name_first = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "张三 销售总监",
                "13800138000",
                "中山路128号",
            ]
        )
        self.assertEqual("张三", cn_name_first["姓名"])
        self.assertEqual("张三 销售总监", cn_name_first["职位"])

        engineer = ocr_engine._map_business_card_fields(
            [
                "上海某某有限公司",
                "软件工程师 李四",
                "13800138000",
                "中山路128号",
            ]
        )
        self.assertEqual("李四", engineer["姓名"])
        self.assertNotEqual("软件", engineer["姓名"])

        director = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Director Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", director["姓名"])
        self.assertEqual("Director Jane Doe", director["职位"])
        self.assertEqual("Acme Inc", director["公司"])

        vp = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "VP Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", vp["姓名"])
        self.assertEqual("VP Jane Doe", vp["职位"])

        cto_line = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Chief Technology Officer",
                "Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", cto_line["姓名"])
        self.assertEqual("Chief Technology Officer", cto_line["职位"])
        self.assertNotEqual("Chief Technology Officer", cto_line["姓名"])

    def test_foundation_college_technologies_are_companies_not_names(self):
        foundation = ocr_engine._map_business_card_fields(
            [
                "Gates Foundation",
                "Jane Doe",
                "Director",
                "500 5th Avenue",
            ]
        )
        self.assertEqual("Jane Doe", foundation["姓名"])
        self.assertEqual("Gates Foundation", foundation["公司"])
        self.assertEqual("Director", foundation["职位"])

        college = ocr_engine._map_business_card_fields(
            [
                "Williams College",
                "Jane Doe",
                "Professor",
                "880 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", college["姓名"])
        self.assertEqual("Williams College", college["公司"])
        self.assertEqual("Professor", college["职位"])

        technologies = ocr_engine._map_business_card_fields(
            [
                "Acme Technologies",
                "Jane Doe",
                "jane@acme.com",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", technologies["姓名"])
        self.assertEqual("Acme Technologies", technologies["公司"])
        self.assertNotEqual("Acme Technologies", technologies["姓名"])

        labs = ocr_engine._map_business_card_fields(
            [
                "DeepMind Labs",
                "Jane Doe",
                "Scientist",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", labs["姓名"])
        self.assertEqual("DeepMind Labs", labs["公司"])
        self.assertEqual("Scientist", labs["职位"])

    def test_city_prefix_company_line_is_not_used_as_name(self):
        result = ocr_engine._map_business_card_fields(
            [
                "杭州阿里云",
                "张三",
                "经理",
                "13800138000",
            ]
        )
        self.assertEqual("张三", result["姓名"])
        self.assertNotEqual("杭州阿里", result["姓名"])

    def test_insurance_securities_factory_pharma_are_companies_not_names(self):
        pingan = ocr_engine._map_business_card_fields(
            [
                "平安保险",
                "李四",
                "业务经理",
                "13800138000",
                "中山路1号",
            ]
        )
        self.assertEqual("李四", pingan["姓名"])
        self.assertEqual("平安保险", pingan["公司"])
        self.assertEqual("业务经理", pingan["职位"])
        self.assertNotEqual("平安保险", pingan["姓名"])

        citic = ocr_engine._map_business_card_fields(
            [
                "中信证券",
                "张三",
                "客户经理",
                "中关村大街1号",
            ]
        )
        self.assertEqual("张三", citic["姓名"])
        self.assertEqual("中信证券", citic["公司"])
        self.assertEqual("客户经理", citic["职位"])
        self.assertNotEqual("中信证券", citic["姓名"])

        factory = ocr_engine._map_business_card_fields(
            [
                "曙光机械厂",
                "王五",
                "厂长",
                "解放路1号",
            ]
        )
        self.assertEqual("王五", factory["姓名"])
        self.assertEqual("曙光机械厂", factory["公司"])
        self.assertEqual("厂长", factory["职位"])
        self.assertNotEqual("曙光机械", factory["姓名"])

        pharma = ocr_engine._map_business_card_fields(
            [
                "恒瑞制药",
                "赵六",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("赵六", pharma["姓名"])
        self.assertEqual("恒瑞制药", pharma["公司"])
        self.assertNotEqual("恒瑞制药", pharma["姓名"])

        fund = ocr_engine._map_business_card_fields(
            [
                "华夏基金",
                "钱七",
                "基金经理",
                "金融街1号",
            ]
        )
        self.assertEqual("钱七", fund["姓名"])
        self.assertEqual("华夏基金", fund["公司"])
        self.assertEqual("基金经理", fund["职位"])

        yaoye = ocr_engine._map_business_card_fields(
            [
                "同仁堂药业",
                "孙八",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("孙八", yaoye["姓名"])
        self.assertEqual("同仁堂药业", yaoye["公司"])
        self.assertNotEqual("同仁堂药", yaoye["姓名"])

        media = ocr_engine._map_business_card_fields(
            [
                "新浪传媒",
                "周九",
                "总监",
                "中山路1号",
            ]
        )
        self.assertEqual("周九", media["姓名"])
        self.assertEqual("新浪传媒", media["公司"])
        self.assertNotEqual("新浪传媒", media["姓名"])

        trade = ocr_engine._map_business_card_fields(
            [
                "中化贸易",
                "吴十",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("吴十", trade["姓名"])
        self.assertEqual("中化贸易", trade["公司"])

        title_not_company = ocr_engine._map_business_card_fields(
            [
                "平安保险",
                "李四",
                "保险顾问",
                "中山路1号",
            ]
        )
        self.assertEqual("李四", title_not_company["姓名"])
        self.assertEqual("平安保险", title_not_company["公司"])
        self.assertEqual("保险顾问", title_not_company["职位"])

        analyst = ocr_engine._map_business_card_fields(
            [
                "中信证券",
                "张三",
                "证券分析师",
                "金融街1号",
            ]
        )
        self.assertEqual("张三", analyst["姓名"])
        self.assertEqual("中信证券", analyst["公司"])
        self.assertEqual("证券分析师", analyst["职位"])

        pharma_eng = ocr_engine._map_business_card_fields(
            [
                "恒瑞制药",
                "赵六",
                "制药工程师",
                "中山路1号",
            ]
        )
        self.assertEqual("赵六", pharma_eng["姓名"])
        self.assertEqual("恒瑞制药", pharma_eng["公司"])
        self.assertEqual("制药工程师", pharma_eng["职位"])

    def test_english_co_company_factory_securities_are_companies_not_names(self):
        for company in ("Acme Co", "Acme Co.", "Acme Company", "Acme Factory"):
            with self.subTest(company=company):
                result = ocr_engine._map_business_card_fields(
                    [
                        company,
                        "Jane Doe",
                        "Director",
                        "100 Main Street",
                    ]
                )
                self.assertEqual("Jane Doe", result["姓名"])
                self.assertEqual(company, result["公司"])
                self.assertEqual("Director", result["职位"])
                self.assertNotEqual(company, result["姓名"])

        securities = ocr_engine._map_business_card_fields(
            [
                "CITIC Securities",
                "Director",
                "Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", securities["姓名"])
        self.assertEqual("CITIC Securities", securities["公司"])
        self.assertEqual("Director", securities["职位"])
        self.assertNotEqual("CITIC Securities", securities["姓名"])

        insurance = ocr_engine._map_business_card_fields(
            [
                "AIA Insurance",
                "Jane Doe",
                "Consultant",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", insurance["姓名"])
        self.assertEqual("AIA Insurance", insurance["公司"])
        self.assertEqual("Consultant", insurance["职位"])

        fund = ocr_engine._map_business_card_fields(
            [
                "Fidelity Fund",
                "Director",
                "Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", fund["姓名"])
        self.assertEqual("Fidelity Fund", fund["公司"])
        self.assertNotEqual("Fidelity Fund", fund["姓名"])

        company_director = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Company Director",
                "Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", company_director["姓名"])
        self.assertEqual("Acme Inc", company_director["公司"])
        self.assertEqual("Company Director", company_director["职位"])

        factory_manager = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Factory Manager",
                "Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", factory_manager["姓名"])
        self.assertEqual("Acme Inc", factory_manager["公司"])
        self.assertEqual("Factory Manager", factory_manager["职位"])

        cofounder = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Co-Founder Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", cofounder["姓名"])
        self.assertEqual("Co-Founder Jane Doe", cofounder["职位"])
        self.assertEqual("Acme Inc", cofounder["公司"])
        self.assertNotEqual("Co Jane Doe", cofounder["姓名"])

    def test_realty_capital_logistics_are_companies_not_names(self):
        vanke = ocr_engine._map_business_card_fields(
            [
                "万科地产",
                "张三",
                "项目经理",
                "中山路1号",
            ]
        )
        self.assertEqual("张三", vanke["姓名"])
        self.assertEqual("万科地产", vanke["公司"])
        self.assertEqual("项目经理", vanke["职位"])
        self.assertNotEqual("万科地产", vanke["姓名"])

        sequoia = ocr_engine._map_business_card_fields(
            [
                "红杉资本",
                "沈南鹏",
                "合伙人",
                "金融街1号",
            ]
        )
        self.assertEqual("沈南鹏", sequoia["姓名"])
        self.assertEqual("红杉资本", sequoia["公司"])
        self.assertEqual("合伙人", sequoia["职位"])
        self.assertNotEqual("红杉资本", sequoia["姓名"])

        hillhouse = ocr_engine._map_business_card_fields(
            [
                "高瓴投资",
                "张磊",
                "创始人",
                "金融街1号",
            ]
        )
        self.assertEqual("张磊", hillhouse["姓名"])
        self.assertEqual("高瓴投资", hillhouse["公司"])
        self.assertEqual("创始人", hillhouse["职位"])

        logistics = ocr_engine._map_business_card_fields(
            [
                "京东物流",
                "李明",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("李明", logistics["姓名"])
        self.assertEqual("京东物流", logistics["公司"])
        self.assertNotEqual("京东物流", logistics["姓名"])

        express = ocr_engine._map_business_card_fields(
            [
                "顺丰快递",
                "王芳",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("王芳", express["姓名"])
        self.assertEqual("顺丰快递", express["公司"])

        consulting = ocr_engine._map_business_card_fields(
            [
                "德勤咨询",
                "赵强",
                "顾问",
                "中山路1号",
            ]
        )
        self.assertEqual("赵强", consulting["姓名"])
        self.assertEqual("德勤咨询", consulting["公司"])
        self.assertEqual("顾问", consulting["职位"])

        construction = ocr_engine._map_business_card_fields(
            [
                "中铁建设",
                "钱伟",
                "工程师",
                "解放路1号",
            ]
        )
        self.assertEqual("钱伟", construction["姓名"])
        self.assertEqual("中铁建设", construction["公司"])
        self.assertEqual("工程师", construction["职位"])

        hotel = ocr_engine._map_business_card_fields(
            [
                "如家酒店",
                "孙八",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("孙八", hotel["姓名"])
        self.assertEqual("如家酒店", hotel["公司"])

        airline = ocr_engine._map_business_card_fields(
            [
                "南方航空",
                "周九",
                "经理",
                "机场路1号",
            ]
        )
        self.assertEqual("周九", airline["姓名"])
        self.assertEqual("南方航空", airline["公司"])

        grid = ocr_engine._map_business_card_fields(
            [
                "国家电网",
                "吴十",
                "工程师",
                "长安街1号",
            ]
        )
        self.assertEqual("吴十", grid["姓名"])
        self.assertEqual("国家电网", grid["公司"])

        auto = ocr_engine._map_business_card_fields(
            [
                "长城汽车",
                "冯二",
                "总监",
                "朝阳路1号",
            ]
        )
        self.assertEqual("冯二", auto["姓名"])
        self.assertEqual("长城汽车", auto["公司"])

        chemical = ocr_engine._map_business_card_fields(
            [
                "万华化工",
                "蒋六",
                "工程师",
                "中山路1号",
            ]
        )
        self.assertEqual("蒋六", chemical["姓名"])
        self.assertEqual("万华化工", chemical["公司"])

        property_co = ocr_engine._map_business_card_fields(
            [
                "华润置地",
                "韩八",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("韩八", property_co["姓名"])
        self.assertEqual("华润置地", property_co["公司"])

        shipping = ocr_engine._map_business_card_fields(
            [
                "中远海运",
                "杨九",
                "经理",
                "外滩1号",
            ]
        )
        self.assertEqual("杨九", shipping["姓名"])
        self.assertEqual("中远海运", shipping["公司"])

        education = ocr_engine._map_business_card_fields(
            [
                "新东方教育",
                "沈七",
                "老师",
                "中山路1号",
            ]
        )
        self.assertEqual("沈七", education["姓名"])
        self.assertEqual("新东方教育", education["公司"])
        self.assertEqual("老师", education["职位"])
        self.assertNotEqual("新东方教", education["姓名"])

        # 弱公司词不能把职位行或街道行当成公司。
        title_not_company = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "李四",
                "投资经理",
                "中山路1号",
            ]
        )
        self.assertEqual("李四", title_not_company["姓名"])
        self.assertEqual("某某科技", title_not_company["公司"])
        self.assertEqual("投资经理", title_not_company["职位"])

        address_not_company = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "李四",
                "经理",
                "建设路88号",
            ]
        )
        self.assertEqual("李四", address_not_company["姓名"])
        self.assertEqual("某某科技", address_not_company["公司"])
        self.assertEqual("建设路88号", address_not_company["地址"])

    def test_assistant_representative_specialist_are_titles_not_names(self):
        assistant = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "行政助理",
                "李娜",
                "中山路1号",
            ]
        )
        self.assertEqual("李娜", assistant["姓名"])
        self.assertEqual("行政助理", assistant["职位"])
        self.assertNotEqual("行政助理", assistant["姓名"])

        mixed_assistant = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "行政助理 李娜",
                "中山路1号",
            ]
        )
        self.assertEqual("李娜", mixed_assistant["姓名"])
        self.assertEqual("行政助理 李娜", mixed_assistant["职位"])
        self.assertNotEqual("行政助理", mixed_assistant["姓名"])

        representative = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "客户代表",
                "王伟",
                "中山路1号",
            ]
        )
        self.assertEqual("王伟", representative["姓名"])
        self.assertEqual("客户代表", representative["职位"])
        self.assertNotEqual("客户代表", representative["姓名"])

        specialist = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "运营专员",
                "周杰",
                "中山路1号",
            ]
        )
        self.assertEqual("周杰", specialist["姓名"])
        self.assertEqual("运营专员", specialist["职位"])

        accountant = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "会计",
                "赵敏",
                "中山路1号",
            ]
        )
        self.assertEqual("赵敏", accountant["姓名"])
        self.assertEqual("会计", accountant["职位"])

        sales_rep = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Sales Representative Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", sales_rep["姓名"])
        self.assertEqual("Sales Representative Jane Doe", sales_rep["职位"])
        self.assertNotEqual("Sales Representative Jane", sales_rep["姓名"])

        coordinator = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Coordinator Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", coordinator["姓名"])
        self.assertEqual("Coordinator Jane Doe", coordinator["职位"])

    def test_english_capital_logistics_energy_are_companies_not_names(self):
        # Name / Company / Title 是常见英文名片排版；公司行挨着职位时不能把公司写成姓名。
        capital = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Capital",
                "Partner",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", capital["姓名"])
        self.assertEqual("Acme Capital", capital["公司"])
        self.assertEqual("Partner", capital["职位"])
        self.assertNotEqual("Acme Capital", capital["姓名"])

        sequoia = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Sequoia Capital",
                "Partner",
            ]
        )
        self.assertEqual("Jane Doe", sequoia["姓名"])
        self.assertEqual("Sequoia Capital", sequoia["公司"])
        self.assertNotEqual("Sequoia Capital", sequoia["姓名"])

        for company, title in (
            ("DHL Logistics", "Manager"),
            ("Tesla Motors", "Director"),
            ("Pacific Energy", "Engineer"),
            ("Vanke Properties", "Manager"),
            ("Hillhouse Investment", "Partner"),
            ("SF Express", "Manager"),
            ("Hilton Hotels", "Manager"),
            ("United Airlines", "Director"),
            ("Acme Consulting", "Consultant"),
            ("China Telecom", "Manager"),
            ("COSCO Shipping", "Manager"),
            ("Dow Chemical", "Engineer"),
            ("Foxconn Electronics", "Manager"),
            ("Acme Communications", "Director"),
            ("CBRE Realty", "Manager"),
            ("Acme Construction", "Engineer"),
            ("Acme Advertising", "Director"),
            ("New Oriental Education", "Teacher"),
        ):
            with self.subTest(company=company):
                result = ocr_engine._map_business_card_fields(
                    [
                        company,
                        "Jane Doe",
                        title,
                        "100 Main Street",
                    ]
                )
                self.assertEqual("Jane Doe", result["姓名"])
                self.assertEqual(company, result["公司"])
                self.assertEqual(title, result["职位"])
                self.assertNotEqual(company, result["姓名"])

        investment_director = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Investment Director",
                "Jane Doe",
                "100 Main Street",
            ]
        )
        self.assertEqual("Jane Doe", investment_director["姓名"])
        self.assertEqual("Acme Inc", investment_director["公司"])
        self.assertEqual("Investment Director", investment_director["职位"])

    def test_life_insurance_trust_medical_retail_are_companies_not_names(self):
        life = ocr_engine._map_business_card_fields(
            [
                "新华人寿",
                "李四",
                "经理",
                "金融街1号",
            ]
        )
        self.assertEqual("李四", life["姓名"])
        self.assertEqual("新华人寿", life["公司"])
        self.assertEqual("经理", life["职位"])
        self.assertNotEqual("新华人寿", life["姓名"])

        china_life = ocr_engine._map_business_card_fields(
            [
                "中国人寿",
                "张三",
                "客户经理",
                "金融街1号",
            ]
        )
        self.assertEqual("张三", china_life["姓名"])
        self.assertEqual("中国人寿", china_life["公司"])
        self.assertEqual("客户经理", china_life["职位"])

        trust = ocr_engine._map_business_card_fields(
            [
                "中信信托",
                "王五",
                "经理",
                "金融街1号",
            ]
        )
        self.assertEqual("王五", trust["姓名"])
        self.assertEqual("中信信托", trust["公司"])
        self.assertNotEqual("中信信托", trust["姓名"])

        futures = ocr_engine._map_business_card_fields(
            [
                "中信期货",
                "赵六",
                "分析师",
                "金融街1号",
            ]
        )
        self.assertEqual("赵六", futures["姓名"])
        self.assertEqual("中信期货", futures["公司"])
        self.assertEqual("分析师", futures["职位"])

        leasing = ocr_engine._map_business_card_fields(
            [
                "远东租赁",
                "钱七",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("钱七", leasing["姓名"])
        self.assertEqual("远东租赁", leasing["公司"])

        medical = ocr_engine._map_business_card_fields(
            [
                "迈瑞医疗",
                "吕五",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("吕五", medical["姓名"])
        self.assertEqual("迈瑞医疗", medical["公司"])
        self.assertNotEqual("迈瑞医疗", medical["姓名"])

        biotech = ocr_engine._map_business_card_fields(
            [
                "药明生物",
                "何四",
                "研究员",
                "中山路1号",
            ]
        )
        self.assertEqual("何四", biotech["姓名"])
        self.assertEqual("药明生物", biotech["公司"])
        self.assertEqual("研究员", biotech["职位"])

        supermarket = ocr_engine._map_business_card_fields(
            [
                "永辉超市",
                "吴十",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("吴十", supermarket["姓名"])
        self.assertEqual("永辉超市", supermarket["公司"])
        self.assertNotEqual("永辉超市", supermarket["姓名"])

        department = ocr_engine._map_business_card_fields(
            [
                "王府井百货",
                "周九",
                "店长",
                "王府井大街1号",
            ]
        )
        self.assertEqual("周九", department["姓名"])
        self.assertEqual("王府井百货", department["公司"])
        self.assertEqual("店长", department["职位"])
        self.assertNotEqual("王府井百", department["姓名"])

        food = ocr_engine._map_business_card_fields(
            [
                "三全食品",
                "冯二",
                "总监",
                "中山路1号",
            ]
        )
        self.assertEqual("冯二", food["姓名"])
        self.assertEqual("三全食品", food["公司"])

        liquor = ocr_engine._map_business_card_fields(
            [
                "茅台酒业",
                "陈三",
                "经理",
                "中山路1号",
            ]
        )
        self.assertEqual("陈三", liquor["姓名"])
        self.assertEqual("茅台酒业", liquor["公司"])

        mining = ocr_engine._map_business_card_fields(
            [
                "紫金矿业",
                "褚四",
                "工程师",
                "中山路1号",
            ]
        )
        self.assertEqual("褚四", mining["姓名"])
        self.assertEqual("紫金矿业", mining["公司"])
        self.assertEqual("工程师", mining["职位"])

        semiconductor = ocr_engine._map_business_card_fields(
            [
                "中芯半导体",
                "许三",
                "工程师",
                "中山路1号",
            ]
        )
        self.assertEqual("许三", semiconductor["姓名"])
        self.assertEqual("中芯半导体", semiconductor["公司"])

        # 弱公司词不能把职位行或街道行当成公司，也不能把职位修饰词当成姓名。
        finance_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "金融经理",
                "李四",
                "中山路1号",
            ]
        )
        self.assertEqual("李四", finance_title["姓名"])
        self.assertEqual("某某科技", finance_title["公司"])
        self.assertEqual("金融经理", finance_title["职位"])
        self.assertNotEqual("金融", finance_title["姓名"])

        medical_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "医疗顾问",
                "李四",
                "中山路1号",
            ]
        )
        self.assertEqual("李四", medical_title["姓名"])
        self.assertEqual("某某科技", medical_title["公司"])
        self.assertEqual("医疗顾问", medical_title["职位"])
        self.assertNotEqual("医疗", medical_title["姓名"])

        food_street = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "李四",
                "经理",
                "食品路88号",
            ]
        )
        self.assertEqual("李四", food_street["姓名"])
        self.assertEqual("某某科技", food_street["公司"])
        self.assertEqual("食品路88号", food_street["地址"])

    def test_ventures_pharma_partners_systems_are_companies_not_names(self):
        ventures = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Ventures",
                "Partner",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", ventures["姓名"])
        self.assertEqual("Acme Ventures", ventures["公司"])
        self.assertEqual("Partner", ventures["职位"])
        self.assertNotEqual("Acme Ventures", ventures["姓名"])

        partners = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Smith Partners",
                "Associate",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", partners["姓名"])
        self.assertEqual("Smith Partners", partners["公司"])
        self.assertEqual("Associate", partners["职位"])
        self.assertNotEqual("Smith Partners", partners["姓名"])

        pharma = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Pharma",
                "Director",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", pharma["姓名"])
        self.assertEqual("Acme Pharma", pharma["公司"])
        self.assertEqual("Director", pharma["职位"])

        systems = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Cisco Systems",
                "Engineer",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", systems["姓名"])
        self.assertEqual("Cisco Systems", systems["公司"])
        self.assertEqual("Engineer", systems["职位"])
        self.assertNotEqual("Cisco Systems", systems["姓名"])

        healthcare = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Healthcare",
                "Director",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", healthcare["姓名"])
        self.assertEqual("Acme Healthcare", healthcare["公司"])

        trading_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Trading Manager",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", trading_title["姓名"])
        self.assertEqual("Acme Inc", trading_title["公司"])
        self.assertEqual("Trading Manager", trading_title["职位"])

        systems_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Systems Engineer",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", systems_title["姓名"])
        self.assertEqual("Acme Inc", systems_title["公司"])
        self.assertEqual("Systems Engineer", systems_title["职位"])

    def test_secretary_supervisor_counsel_are_titles_not_names(self):
        secretary = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "党委书记",
                "张三",
                "中山路1号",
            ]
        )
        self.assertEqual("张三", secretary["姓名"])
        self.assertEqual("党委书记", secretary["职位"])
        self.assertNotEqual("党委书记", secretary["姓名"])

        mixed_secretary = ocr_engine._map_business_card_fields(
            [
                "党委书记 张三",
                "某某科技",
                "中山路1号",
            ]
        )
        self.assertEqual("张三", mixed_secretary["姓名"])
        self.assertEqual("党委书记 张三", mixed_secretary["职位"])

        section_chief = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "财务处长",
                "王五",
                "中山路1号",
            ]
        )
        self.assertEqual("王五", section_chief["姓名"])
        self.assertEqual("财务处长", section_chief["职位"])
        self.assertNotEqual("财务处长", section_chief["姓名"])

        store_manager = ocr_engine._map_business_card_fields(
            [
                "永辉超市",
                "店长",
                "吴十",
                "中山路1号",
            ]
        )
        self.assertEqual("吴十", store_manager["姓名"])
        self.assertEqual("永辉超市", store_manager["公司"])
        self.assertEqual("店长", store_manager["职位"])
        self.assertNotEqual("店长", store_manager["姓名"])

        mixed_store = ocr_engine._map_business_card_fields(
            [
                "店长 王芳",
                "永辉超市",
                "中山路1号",
            ]
        )
        self.assertEqual("王芳", mixed_store["姓名"])
        self.assertEqual("店长 王芳", mixed_store["职位"])
        self.assertEqual("永辉超市", mixed_store["公司"])

        nurse = ocr_engine._map_business_card_fields(
            [
                "瑞金医院",
                "护士",
                "陈三",
                "瑞金路1号",
            ]
        )
        self.assertEqual("陈三", nurse["姓名"])
        self.assertEqual("瑞金医院", nurse["公司"])
        self.assertEqual("护士", nurse["职位"])
        self.assertNotEqual("护士", nurse["姓名"])

        office_secretary = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Office Secretary",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", office_secretary["姓名"])
        self.assertEqual("Office Secretary", office_secretary["职位"])
        self.assertNotEqual("Office Secretary", office_secretary["姓名"])

        mixed_en_secretary = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Secretary Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", mixed_en_secretary["姓名"])
        self.assertEqual("Secretary Jane Doe", mixed_en_secretary["职位"])

        supervisor = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Sales Supervisor",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", supervisor["姓名"])
        self.assertEqual("Sales Supervisor", supervisor["职位"])

        counsel = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Legal Counsel Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", counsel["姓名"])
        self.assertEqual("Legal Counsel Jane Doe", counsel["职位"])
        self.assertNotEqual("Legal Counsel Jane", counsel["姓名"])

        developer = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Software Developer",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", developer["姓名"])
        self.assertEqual("Software Developer", developer["职位"])
        self.assertNotEqual("Software Developer", developer["姓名"])

    def test_engineering_machinery_retail_are_companies_not_names(self):
        engineering = ocr_engine._map_business_card_fields(
            [
                "海油工程",
                "张伟",
                "项目经理",
                "上海市浦东新区世纪大道1号",
            ]
        )
        self.assertEqual("张伟", engineering["姓名"])
        self.assertEqual("海油工程", engineering["公司"])
        self.assertEqual("项目经理", engineering["职位"])
        self.assertNotEqual("海油工程", engineering["姓名"])

        heavy = ocr_engine._map_business_card_fields(
            [
                "三一重工",
                "李强",
                "销售总监",
                "长沙市经济开发区",
            ]
        )
        self.assertEqual("李强", heavy["姓名"])
        self.assertEqual("三一重工", heavy["公司"])
        self.assertEqual("销售总监", heavy["职位"])
        self.assertNotEqual("三一重工", heavy["姓名"])

        machinery = ocr_engine._map_business_card_fields(
            [
                "徐工机械",
                "王芳",
                "区域经理",
                "徐州市和平路88号",
            ]
        )
        self.assertEqual("王芳", machinery["姓名"])
        self.assertEqual("徐工机械", machinery["公司"])
        self.assertEqual("区域经理", machinery["职位"])

        appliance = ocr_engine._map_business_card_fields(
            [
                "格力电器",
                "陈明",
                "销售经理",
                "珠海市香洲区",
            ]
        )
        self.assertEqual("陈明", appliance["姓名"])
        self.assertEqual("格力电器", appliance["公司"])
        self.assertNotEqual("格力电器", appliance["姓名"])

        dairy = ocr_engine._map_business_card_fields(
            [
                "伊利乳业",
                "赵敏",
                "市场总监",
                "呼和浩特市金山大道",
            ]
        )
        self.assertEqual("赵敏", dairy["姓名"])
        self.assertEqual("伊利乳业", dairy["公司"])

        sports = ocr_engine._map_business_card_fields(
            [
                "安踏体育",
                "刘洋",
                "品牌经理",
                "晋江市池店镇",
            ]
        )
        self.assertEqual("刘洋", sports["姓名"])
        self.assertEqual("安踏体育", sports["公司"])
        self.assertNotEqual("安踏体育", sports["姓名"])

        film = ocr_engine._map_business_card_fields(
            [
                "万达影业",
                "孙丽",
                "制片主任",
                "北京市朝阳区",
            ]
        )
        self.assertEqual("孙丽", film["姓名"])
        self.assertEqual("万达影业", film["公司"])
        self.assertEqual("制片主任", film["职位"])

        travel = ocr_engine._map_business_card_fields(
            [
                "携程旅游",
                "周杰",
                "运营经理",
                "上海市长宁区",
            ]
        )
        self.assertEqual("周杰", travel["姓名"])
        self.assertEqual("携程旅游", travel["公司"])

        apparel = ocr_engine._map_business_card_fields(
            [
                "波司登服装",
                "吴倩",
                "设计总监",
                "常熟市虞山镇",
            ]
        )
        self.assertEqual("吴倩", apparel["姓名"])
        self.assertEqual("波司登服装", apparel["公司"])
        self.assertNotEqual("波司登服", apparel["姓名"])

        textile = ocr_engine._map_business_card_fields(
            [
                "鲁泰纺织",
                "郑华",
                "厂长",
                "淄博市高新区",
            ]
        )
        self.assertEqual("郑华", textile["姓名"])
        self.assertEqual("鲁泰纺织", textile["公司"])
        self.assertEqual("厂长", textile["职位"])

        building = ocr_engine._map_business_card_fields(
            [
                "北新建材",
                "马超",
                "工程师",
                "北京市海淀区",
            ]
        )
        self.assertEqual("马超", building["姓名"])
        self.assertEqual("北新建材", building["公司"])

        cement = ocr_engine._map_business_card_fields(
            [
                "海螺水泥",
                "黄磊",
                "生产经理",
                "芜湖市弋江区",
            ]
        )
        self.assertEqual("黄磊", cement["姓名"])
        self.assertEqual("海螺水泥", cement["公司"])

        glass = ocr_engine._map_business_card_fields(
            [
                "福耀玻璃",
                "林涛",
                "质量主管",
                "福清市宏路镇",
            ]
        )
        self.assertEqual("林涛", glass["姓名"])
        self.assertEqual("福耀玻璃", glass["公司"])

        ship = ocr_engine._map_business_card_fields(
            [
                "中国船舶",
                "高峰",
                "技术总监",
                "上海市浦东新区",
            ]
        )
        self.assertEqual("高峰", ship["姓名"])
        self.assertEqual("中国船舶", ship["公司"])

        solar = ocr_engine._map_business_card_fields(
            [
                "隆基光伏",
                "何静",
                "销售经理",
                "西安市长安区",
            ]
        )
        self.assertEqual("何静", solar["姓名"])
        self.assertEqual("隆基光伏", solar["公司"])

        pharmacy = ocr_engine._map_business_card_fields(
            [
                "益丰药房",
                "唐雪",
                "店长",
                "长沙市岳麓区",
            ]
        )
        self.assertEqual("唐雪", pharmacy["姓名"])
        self.assertEqual("益丰药房", pharmacy["公司"])
        self.assertEqual("店长", pharmacy["职位"])

        furniture = ocr_engine._map_business_card_fields(
            [
                "索菲亚家居",
                "冯伟",
                "设计师",
                "广州市增城区",
            ]
        )
        self.assertEqual("冯伟", furniture["姓名"])
        self.assertEqual("索菲亚家居", furniture["公司"])
        self.assertNotEqual("索菲亚家", furniture["姓名"])

        paint = ocr_engine._map_business_card_fields(
            [
                "三棵树涂料",
                "曹阳",
                "销售代表",
                "莆田市荔城区",
            ]
        )
        self.assertEqual("曹阳", paint["姓名"])
        self.assertEqual("三棵树涂料", paint["公司"])
        self.assertNotEqual("三棵树涂", paint["姓名"])

        warehouse = ocr_engine._map_business_card_fields(
            [
                "中储仓储",
                "沈杰",
                "物流经理",
                "天津市北辰区",
            ]
        )
        self.assertEqual("沈杰", warehouse["姓名"])
        self.assertEqual("中储仓储", warehouse["公司"])

        construction = ocr_engine._map_business_card_fields(
            [
                "上海建工",
                "丁磊",
                "工程师",
                "上海市静安区",
            ]
        )
        self.assertEqual("丁磊", construction["姓名"])
        self.assertEqual("上海建工", construction["公司"])

        museum = ocr_engine._map_business_card_fields(
            [
                "上海博物馆",
                "馆长",
                "郑华",
                "人民大道201号",
            ]
        )
        self.assertEqual("郑华", museum["姓名"])
        self.assertEqual("上海博物馆", museum["公司"])
        self.assertEqual("馆长", museum["职位"])
        self.assertNotEqual("馆长", museum["姓名"])

        # 弱公司词不能把职位行或街道行当成公司，也不能把职位修饰词当成姓名。
        machinery_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "机械工程师",
                "张伟",
                "中山路1号",
            ]
        )
        self.assertEqual("张伟", machinery_title["姓名"])
        self.assertEqual("某某科技", machinery_title["公司"])
        self.assertEqual("机械工程师", machinery_title["职位"])
        self.assertNotEqual("机械", machinery_title["姓名"])

        electric_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "电器工程师",
                "张伟",
                "中山路1号",
            ]
        )
        self.assertEqual("张伟", electric_title["姓名"])
        self.assertEqual("电器工程师", electric_title["职位"])
        self.assertNotEqual("电器", electric_title["姓名"])

        sports_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "体育老师",
                "张伟",
                "中山路1号",
            ]
        )
        self.assertEqual("张伟", sports_title["姓名"])
        self.assertEqual("体育老师", sports_title["职位"])
        self.assertNotEqual("体育", sports_title["姓名"])

        apparel_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "服装设计师",
                "张伟",
                "中山路1号",
            ]
        )
        self.assertEqual("张伟", apparel_title["姓名"])
        self.assertEqual("服装设计师", apparel_title["职位"])
        self.assertNotEqual("服装", apparel_title["姓名"])

        engineering_title = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "工程经理",
                "张伟",
                "中山路1号",
            ]
        )
        self.assertEqual("张伟", engineering_title["姓名"])
        self.assertEqual("工程经理", engineering_title["职位"])
        self.assertNotEqual("工程", engineering_title["姓名"])

        engineering_street = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "张伟",
                "项目经理",
                "工程路88号",
            ]
        )
        self.assertEqual("张伟", engineering_street["姓名"])
        self.assertEqual("某某科技", engineering_street["公司"])
        self.assertEqual("工程路88号", engineering_street["地址"])

        machinery_street = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "张伟",
                "区域经理",
                "机械路88号",
            ]
        )
        self.assertEqual("张伟", machinery_street["姓名"])
        self.assertEqual("某某科技", machinery_street["公司"])
        self.assertEqual("机械路88号", machinery_street["地址"])

        sports_street = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "张伟",
                "品牌经理",
                "体育路88号",
            ]
        )
        self.assertEqual("张伟", sports_street["姓名"])
        self.assertEqual("某某科技", sports_street["公司"])
        self.assertEqual("体育路88号", sports_street["地址"])

        paint_street = ocr_engine._map_business_card_fields(
            [
                "某某科技",
                "张伟",
                "销售经理",
                "涂料路88号",
            ]
        )
        self.assertEqual("张伟", paint_street["姓名"])
        self.assertEqual("某某科技", paint_street["公司"])
        self.assertEqual("涂料路88号", paint_street["地址"])

    def test_engineering_associates_software_are_companies_not_names(self):
        engineering = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Engineering",
                "Director",
                "88 West Rd",
            ]
        )
        self.assertEqual("Jane Doe", engineering["姓名"])
        self.assertEqual("Acme Engineering", engineering["公司"])
        self.assertEqual("Director", engineering["职位"])
        self.assertNotEqual("Acme Engineering", engineering["姓名"])

        engineering_first = ocr_engine._map_business_card_fields(
            [
                "Acme Engineering",
                "Jane Doe",
                "Director",
                "88 West Rd",
            ]
        )
        self.assertEqual("Jane Doe", engineering_first["姓名"])
        self.assertEqual("Acme Engineering", engineering_first["公司"])

        machinery = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Machinery",
                "Director",
            ]
        )
        self.assertEqual("Jane Doe", machinery["姓名"])
        self.assertEqual("Acme Machinery", machinery["公司"])
        self.assertEqual("Director", machinery["职位"])

        industries = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Industries",
                "Director",
            ]
        )
        self.assertEqual("Jane Doe", industries["姓名"])
        self.assertEqual("Acme Industries", industries["公司"])
        self.assertNotEqual("Acme Industries", industries["姓名"])

        manufacturing = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Manufacturing",
                "Director",
            ]
        )
        self.assertEqual("Jane Doe", manufacturing["姓名"])
        self.assertEqual("Acme Manufacturing", manufacturing["公司"])
        self.assertNotEqual("Acme Manufacturing", manufacturing["姓名"])

        associates = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Wilson Associates",
                "Counsel",
            ]
        )
        self.assertEqual("Jane Doe", associates["姓名"])
        self.assertEqual("Wilson Associates", associates["公司"])
        self.assertEqual("Counsel", associates["职位"])
        self.assertNotEqual("Wilson Associates", associates["姓名"])

        advisors = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Advisors",
                "Partner",
            ]
        )
        self.assertEqual("Jane Doe", advisors["姓名"])
        self.assertEqual("Acme Advisors", advisors["公司"])
        self.assertEqual("Partner", advisors["职位"])

        international = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme International",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", international["姓名"])
        self.assertEqual("Acme International", international["公司"])
        self.assertNotEqual("Acme International", international["姓名"])

        global_co = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Global",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", global_co["姓名"])
        self.assertEqual("Acme Global", global_co["公司"])
        self.assertNotEqual("Acme Global", global_co["姓名"])

        robotics = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Robotics",
                "Engineer",
            ]
        )
        self.assertEqual("Jane Doe", robotics["姓名"])
        self.assertEqual("Acme Robotics", robotics["公司"])

        networks = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Networks",
                "Engineer",
            ]
        )
        self.assertEqual("Jane Doe", networks["姓名"])
        self.assertEqual("Acme Networks", networks["公司"])

        software = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Software",
                "Engineer",
            ]
        )
        self.assertEqual("Jane Doe", software["姓名"])
        self.assertEqual("Acme Software", software["公司"])
        self.assertNotEqual("Acme Software", software["姓名"])

        apparel = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Apparel",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", apparel["姓名"])
        self.assertEqual("Acme Apparel", apparel["公司"])

        retail = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Retail",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", retail["姓名"])
        self.assertEqual("Acme Retail", retail["公司"])

        beverages = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Beverages",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", beverages["姓名"])
        self.assertEqual("Acme Beverages", beverages["公司"])

        materials = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Materials",
                "Engineer",
            ]
        )
        self.assertEqual("Jane Doe", materials["姓名"])
        self.assertEqual("Acme Materials", materials["公司"])

        equipment = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Equipment",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", equipment["姓名"])
        self.assertEqual("Acme Equipment", equipment["公司"])

        packaging = ocr_engine._map_business_card_fields(
            [
                "Jane Doe",
                "Acme Packaging",
                "Manager",
            ]
        )
        self.assertEqual("Jane Doe", packaging["姓名"])
        self.assertEqual("Acme Packaging", packaging["公司"])

        engineering_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Engineering Manager",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", engineering_title["姓名"])
        self.assertEqual("Acme Inc", engineering_title["公司"])
        self.assertEqual("Engineering Manager", engineering_title["职位"])
        self.assertNotEqual("Engineering Manager", engineering_title["姓名"])

        machinery_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Machinery Manager",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", machinery_title["姓名"])
        self.assertEqual("Machinery Manager", machinery_title["职位"])

        industries_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Industries Director",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", industries_title["姓名"])
        self.assertEqual("Industries Director", industries_title["职位"])

        software_title = ocr_engine._map_business_card_fields(
            [
                "软件工程师 Jane Doe",
                "Acme Inc",
            ]
        )
        self.assertEqual("Jane Doe", software_title["姓名"])
        self.assertEqual("Acme Inc", software_title["公司"])
        self.assertEqual("软件工程师 Jane Doe", software_title["职位"])

        retail_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Retail Manager",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", retail_title["姓名"])
        self.assertEqual("Acme Inc", retail_title["公司"])
        self.assertEqual("Retail Manager", retail_title["职位"])

        global_title = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Global Manager",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", global_title["姓名"])
        self.assertEqual("Acme Inc", global_title["公司"])
        self.assertEqual("Global Manager", global_title["职位"])

    def test_driver_chef_photographer_are_titles_not_names(self):
        driver = ocr_engine._map_business_card_fields(
            [
                "顺丰速运",
                "司机",
                "张伟",
                "13800138000",
            ]
        )
        self.assertEqual("张伟", driver["姓名"])
        self.assertEqual("顺丰速运", driver["公司"])
        self.assertEqual("司机", driver["职位"])
        self.assertNotEqual("司机", driver["姓名"])

        mixed_driver = ocr_engine._map_business_card_fields(
            [
                "司机 张伟",
                "顺丰速运",
                "13800138000",
            ]
        )
        self.assertEqual("张伟", mixed_driver["姓名"])
        self.assertEqual("司机 张伟", mixed_driver["职位"])
        self.assertEqual("顺丰速运", mixed_driver["公司"])

        chef = ocr_engine._map_business_card_fields(
            [
                "海底捞餐饮",
                "厨师",
                "李强",
                "13800138001",
            ]
        )
        self.assertEqual("李强", chef["姓名"])
        self.assertEqual("海底捞餐饮", chef["公司"])
        self.assertEqual("厨师", chef["职位"])
        self.assertNotEqual("厨师", chef["姓名"])

        security = ocr_engine._map_business_card_fields(
            [
                "万科物业",
                "保安",
                "王刚",
                "13800138002",
            ]
        )
        self.assertEqual("王刚", security["姓名"])
        self.assertEqual("万科物业", security["公司"])
        self.assertEqual("保安", security["职位"])

        support = ocr_engine._map_business_card_fields(
            [
                "京东物流",
                "客服",
                "赵敏",
                "13800138003",
            ]
        )
        self.assertEqual("赵敏", support["姓名"])
        self.assertEqual("京东物流", support["公司"])
        self.assertEqual("客服", support["职位"])
        self.assertNotEqual("客服", support["姓名"])

        buyer = ocr_engine._map_business_card_fields(
            [
                "格力电器",
                "采购",
                "刘洋",
                "13800138004",
            ]
        )
        self.assertEqual("刘洋", buyer["姓名"])
        self.assertEqual("格力电器", buyer["公司"])
        self.assertEqual("采购", buyer["职位"])
        self.assertNotEqual("苏宁电器", buyer["姓名"])

        auditor = ocr_engine._map_business_card_fields(
            [
                "立信会计师事务所",
                "审计",
                "陈芳",
                "13800138005",
            ]
        )
        self.assertEqual("陈芳", auditor["姓名"])
        self.assertEqual("立信会计师事务所", auditor["公司"])
        self.assertEqual("审计", auditor["职位"])
        self.assertNotEqual("审计", auditor["姓名"])

        programmer = ocr_engine._map_business_card_fields(
            [
                "杭州云启科技",
                "程序员",
                "周杰",
                "13800138006",
            ]
        )
        self.assertEqual("周杰", programmer["姓名"])
        self.assertEqual("杭州云启科技", programmer["公司"])
        self.assertEqual("程序员", programmer["职位"])
        self.assertNotEqual("程序员", programmer["姓名"])

        guide = ocr_engine._map_business_card_fields(
            [
                "携程旅游",
                "导游",
                "吴倩",
                "13800138007",
            ]
        )
        self.assertEqual("吴倩", guide["姓名"])
        self.assertEqual("携程旅游", guide["公司"])
        self.assertEqual("导游", guide["职位"])

        photographer = ocr_engine._map_business_card_fields(
            [
                "光线传媒",
                "摄影师",
                "马超",
                "13800138009",
            ]
        )
        self.assertEqual("马超", photographer["姓名"])
        self.assertEqual("光线传媒", photographer["公司"])
        self.assertEqual("摄影师", photographer["职位"])
        self.assertNotEqual("摄影师", photographer["姓名"])

        coach = ocr_engine._map_business_card_fields(
            [
                "安踏体育",
                "教练",
                "黄磊",
                "13800138010",
            ]
        )
        self.assertEqual("黄磊", coach["姓名"])
        self.assertEqual("安踏体育", coach["公司"])
        self.assertEqual("教练", coach["职位"])

        actuary = ocr_engine._map_business_card_fields(
            [
                "中国人寿",
                "精算师",
                "高峰",
                "13800138012",
            ]
        )
        self.assertEqual("高峰", actuary["姓名"])
        self.assertEqual("中国人寿", actuary["公司"])
        self.assertEqual("精算师", actuary["职位"])
        self.assertNotEqual("精算师", actuary["姓名"])

        warehouse = ocr_engine._map_business_card_fields(
            [
                "京东物流",
                "仓管",
                "何静",
                "13800138013",
            ]
        )
        self.assertEqual("何静", warehouse["姓名"])
        self.assertEqual("仓管", warehouse["职位"])
        self.assertNotEqual("仓管", warehouse["姓名"])

        cashier = ocr_engine._map_business_card_fields(
            [
                "永辉超市",
                "收银",
                "许飞",
                "中山路1号",
            ]
        )
        self.assertEqual("许飞", cashier["姓名"])
        self.assertEqual("永辉超市", cashier["公司"])
        self.assertEqual("收银", cashier["职位"])
        self.assertEqual("中山路1号", cashier["地址"])
        self.assertNotEqual("收银", cashier["姓名"])

        clerk = ocr_engine._map_business_card_fields(
            [
                "王府井百货",
                "导购",
                "唐雪",
                "13800138015",
            ]
        )
        self.assertEqual("唐雪", clerk["姓名"])
        self.assertEqual("导购", clerk["职位"])
        self.assertNotEqual("导购", clerk["姓名"])

        translator = ocr_engine._map_business_card_fields(
            [
                "新东方教育",
                "翻译",
                "冯伟",
                "13800138016",
            ]
        )
        self.assertEqual("冯伟", translator["姓名"])
        self.assertEqual("新东方教育", translator["公司"])
        self.assertEqual("翻译", translator["职位"])

        trainer = ocr_engine._map_business_card_fields(
            [
                "新东方教育",
                "培训师",
                "曹阳",
                "13800138017",
            ]
        )
        self.assertEqual("曹阳", trainer["姓名"])
        self.assertEqual("培训师", trainer["职位"])
        self.assertNotEqual("培训师", trainer["姓名"])

        chef_en = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Chef Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", chef_en["姓名"])
        self.assertEqual("Chef Jane Doe", chef_en["职位"])
        self.assertEqual("Acme Inc", chef_en["公司"])

        photographer_en = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Photographer",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", photographer_en["姓名"])
        self.assertEqual("Photographer", photographer_en["职位"])
        self.assertNotEqual("Photographer", photographer_en["姓名"])

        auditor_en = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Auditor Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", auditor_en["姓名"])
        self.assertEqual("Auditor Jane Doe", auditor_en["职位"])

        programmer_en = ocr_engine._map_business_card_fields(
            [
                "Acme Inc",
                "Programmer",
                "Jane Doe",
                "1 Main St",
            ]
        )
        self.assertEqual("Jane Doe", programmer_en["姓名"])
        self.assertEqual("Programmer", programmer_en["职位"])


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
