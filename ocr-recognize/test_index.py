import json
import unittest
from unittest.mock import patch

import index
from config import MAX_IMAGE_SIZE


class HandlerSizeLimitTest(unittest.TestCase):
    def test_rejects_oversized_image_before_calling_model(self):
        event = {
            "body": json.dumps(
                {
                    "image_base64": "A" * (MAX_IMAGE_SIZE + 1),
                    "template_type": "business_card",
                }
            )
        }

        with patch("index.call_llm", side_effect=AssertionError("model should not be called")):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 413)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])
        self.assertIn("图片数据过大", body["error"])

    def test_rejects_very_large_body_before_json_parse(self):
        event = {"body": "{" + ("x" * (MAX_IMAGE_SIZE + index.REQUEST_BODY_SIZE_SLACK + 1))}

        with patch("index.call_llm", side_effect=AssertionError("model should not be called")):
            response = index.handler(event, None)

        self.assertEqual(response["statusCode"], 413)
        body = json.loads(response["body"])
        self.assertFalse(body["success"])
        self.assertIn("图片数据过大", body["error"])


if __name__ == "__main__":
    unittest.main()
