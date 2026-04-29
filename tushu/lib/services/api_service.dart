import 'dart:convert';
import 'package:dio/dio.dart';
import '../models/template.dart';

class ApiService {
  // 替换为你的云函数URL
  static const String _baseUrl = "https://ocr-recognize-glkamxdxqt.cn-shanghai.fcapp.run";

  static final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 30),
  ));

  /// 调用后端OCR识别
  static Future<Map<String, String>> recognize({
    required String imageBase64,
    required TemplateType templateType,
    required String customFields,
  }) async {
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

    final response = await _dio.post(_baseUrl, data: requestData);

    if (response.statusCode == 200 && response.data["success"] == true) {
      // 将JSON map转为 String map
      final Map<String, dynamic> data = response.data["data"];
      return data.map((key, value) => MapEntry(key, value.toString()));
    } else {
      throw Exception(response.data["error"] ?? "识别失败");
    }
  }
}