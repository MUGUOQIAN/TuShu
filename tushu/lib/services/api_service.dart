import 'dart:async';
import 'dart:convert';
import 'package:dio/dio.dart';
import '../models/template.dart';

/// OCR 云函数（默认 [ocrBaseUrl] 为已部署的生产地址；本地可覆盖）。
class ApiService {
  /// 生产环境：与 `ocr-recognize` 部署的 HTTP 触发器根 URL 一致。
  static const String _defaultBaseUrl =
      "https://ocr-recognize-glkamxdxqt.cn-shanghai.fcapp.run";

  /// 联调本地后端示例：
  /// `flutter run --dart-define=OCR_API_BASE_URL=http://127.0.0.1:9000`
  static const String _baseUrl = String.fromEnvironment(
    'OCR_API_BASE_URL',
    defaultValue: _defaultBaseUrl,
  );

  /// 实际请求地址（与编译期 `OCR_API_BASE_URL` 或默认生产地址一致）。
  static String get ocrBaseUrl => _baseUrl;

  static final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 60),
    sendTimeout: const Duration(seconds: 120),
    receiveTimeout: const Duration(seconds: 120),
    headers: {'Content-Type': 'application/json; charset=utf-8'},
  ));

  /// 调用后端OCR识别
  static Future<Map<String, String>> recognize({
    required String imageBase64,
    required TemplateType templateType,
    required String customFields,
    Duration requestTimeout = const Duration(seconds: 45),
    Duration? sendTimeout,
    Duration? connectTimeout,
    Duration? receiveTimeout,
  }) async {
    final stopwatch = Stopwatch()..start();
    final requestData = {
      "image_base64": imageBase64,
    };

    if (templateType == TemplateType.custom) {
      requestData["template_type"] = "custom";
      requestData["custom_fields"] = customFields;
    } else if (templateType == TemplateType.businessCard) {
      requestData["template_type"] = "business_card";
    } else if (templateType == TemplateType.invoice) {
      requestData["template_type"] = "invoice";
    }

    print('[ApiService] recognize start -> url=$_baseUrl, template=${requestData["template_type"]}, imageLength=${imageBase64.length}');

    try {
      final response = await _dio
          .post(
            _baseUrl,
            data: requestData,
            options: Options(
              sendTimeout: sendTimeout,
              connectTimeout: connectTimeout,
              receiveTimeout: receiveTimeout,
            ),
          )
          .timeout(requestTimeout);
      print('[ApiService] recognize response <- status=${response.statusCode}, elapsedMs=${stopwatch.elapsedMilliseconds}');

      if (response.statusCode == 200 && response.data["success"] == true) {
        // 将JSON map转为 String map
        final Map<String, dynamic> data = response.data["data"];
        return data.map((key, value) => MapEntry(key, value.toString()));
      } else {
        final responseDataPreview = response.data is String
            ? _preview(response.data as String)
            : _preview(jsonEncode(response.data));
        print('[ApiService] recognize failed payload <- $responseDataPreview');
        throw Exception(response.data["error"] ?? "识别失败");
      }
    } on TimeoutException catch (e) {
      print('[ApiService] recognize timeout after ${stopwatch.elapsedMilliseconds}ms: $e');
      rethrow;
    } on DioException catch (e) {
      final body = e.response?.data;
      final bodyPreview = body == null ? '<empty>' : _preview(body is String ? body : jsonEncode(body));
      print('[ApiService] dio exception type=${e.type}, status=${e.response?.statusCode}, elapsedMs=${stopwatch.elapsedMilliseconds}, body=$bodyPreview');
      rethrow;
    } catch (e) {
      print('[ApiService] unexpected exception after ${stopwatch.elapsedMilliseconds}ms: $e');
      rethrow;
    } finally {
      stopwatch.stop();
    }
  }

  static String _preview(String text, {int max = 300}) {
    if (text.length <= max) {
      return text;
    }
    return '${text.substring(0, max)}...(truncated)';
  }
}
