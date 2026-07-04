import plistlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MobileConfigTests(unittest.TestCase):
    def test_android_main_manifest_declares_required_permissions(self):
        manifest_path = REPO_ROOT / "tushu/android/app/src/main/AndroidManifest.xml"
        root = ET.parse(manifest_path).getroot()
        permissions = {
            item.attrib["{http://schemas.android.com/apk/res/android}name"]
            for item in root.findall("uses-permission")
        }

        self.assertIn("android.permission.INTERNET", permissions)
        self.assertIn("android.permission.CAMERA", permissions)

    def test_ios_info_plist_declares_camera_and_photo_usage(self):
        plist_path = REPO_ROOT / "tushu/ios/Runner/Info.plist"
        with plist_path.open("rb") as fh:
            config = plistlib.load(fh)

        self.assertTrue(config.get("NSCameraUsageDescription"))
        self.assertTrue(config.get("NSPhotoLibraryUsageDescription"))


if __name__ == "__main__":
    unittest.main()
