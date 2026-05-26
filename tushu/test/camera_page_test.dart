import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/models/template.dart';
import 'package:tushu/pages/camera_page.dart';

void main() {
  test('camera page defaults to a usable built-in template', () {
    expect(CameraPage.defaultTemplate, TemplateType.businessCard);
  });
}
