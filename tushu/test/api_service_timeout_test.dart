import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/services/api_service.dart';

void main() {
  test('default request timeout matches the backend receive timeout', () {
    expect(ApiService.defaultRequestTimeout, const Duration(seconds: 120));
  });
}
