import json
import unittest
from unittest.mock import patch

from index import handler


class IndexHandlerTest(unittest.TestCase):
    @patch("index.call_llm")
    def test_handler_passes_expected_fields_to_ocr_engine(self, mock_call_llm):
        mock_call_llm.return_value = {"发票号码": "12345678"}
        event = {
            "body": json.dumps(
                {
                    "image_base64": "abc",
                    "template_type": "invoice",
                },
                ensure_ascii=False,
            )
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["发票号码"], "12345678")
        _, kwargs = mock_call_llm.call_args
        self.assertEqual(
            kwargs["expected_fields"],
            ["发票号码", "开票日期", "购买方名称", "销售方名称", "价税合计金额", "税额"],
        )


if __name__ == "__main__":
    unittest.main()
