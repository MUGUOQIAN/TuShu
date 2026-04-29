import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/services/API_service.dart';
import 'package:tushu/models/template.dart';

void main() {
  test('API recognize businessCard with real image', () async {
    // 读取图片Base64
    final imageBase64 = '''
<BASE64_PLACEHOLDER>
''';
    try {
      final result = await ApiService.recognize(
        imageBase64: imageBase64,
        templateType: TemplateType.businessCard,
        customFields: '',
      );
      print('API 返回: $result');
      expect(result, isA<Map<String, String>>());
    } catch (e) {
      print('API 识别异常: $e');
      fail('API 识别异常: $e');
    }
  });
}
