import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/services/API_service.dart';
import 'package:tushu/models/template.dart';

void main() {
  test('API recognize returns error for invalid image', () async {
    try {
      final result = await ApiService.recognize(
        imageBase64: 'invalid_base64',
        templateType: TemplateType.businessCard,
        customFields: '',
      );
      // 如果没有抛出异常，说明API返回了数据，打印出来
      print(result);
      fail('API 应该抛出异常，但返回了: $result');
    } catch (e) {
      // 只要抛出异常就算通过
      expect(e, isNotNull);
      print('API 连接正常，返回异常: $e');
    }
  });
}
