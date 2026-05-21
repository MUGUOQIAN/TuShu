import 'package:flutter_test/flutter_test.dart';
import 'package:tushu/services/api_service.dart';

void main() {
  test('default OCR request timeout covers upload and receive windows', () {
    expect(
      ApiService.defaultRequestTimeout,
      greaterThanOrEqualTo(const Duration(seconds: 120)),
    );
  });
}
