import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tushu/models/template.dart';
import 'package:tushu/services/API_service.dart';

void main() {
  test('API recognize businessCard with real image', () async {
    final imageFile = File('test/22222.jpg');
    expect(imageFile.existsSync(), isTrue, reason: '缺少测试图片: test/22222.jpg');

    final originalBytes = imageFile.readAsBytesSync();
    final compressedBytes = _compressForUpload(originalBytes);
    final imageBase64 = base64Encode(compressedBytes);
    print(
      '图片压缩: 原始=${originalBytes.length} bytes, 压缩后=${compressedBytes.length} bytes, base64长度=${imageBase64.length}',
    );

    try {
      final result = await ApiService.recognize(
        imageBase64: imageBase64,
        templateType: TemplateType.businessCard,
        customFields: '',
        sendTimeout: const Duration(minutes: 3),
        requestTimeout: const Duration(minutes: 4),
      );
      print('API 返回: $result');
      expect(result, isA<Map<String, String>>());
    } catch (e) {
      print('API 识别异常: $e');
      fail('API 识别异常: $e');
    }
  }, timeout: const Timeout(Duration(minutes: 5)));
}

List<int> _compressForUpload(Uint8List originalBytes) {
  final decoded = img.decodeImage(originalBytes);
  if (decoded == null) {
    throw Exception('无法解码 test/22222.jpg');
  }

  // 将长边限制在 1280，显著减少上传体积。
  final resized = decoded.width >= decoded.height
      ? img.copyResize(decoded, width: 1280)
      : img.copyResize(decoded, height: 1280);

  var quality = 75;
  var jpgBytes = img.encodeJpg(resized, quality: quality);
  while (jpgBytes.length > 450 * 1024 && quality > 35) {
    quality -= 10;
    jpgBytes = img.encodeJpg(resized, quality: quality);
  }

  return jpgBytes;
}
