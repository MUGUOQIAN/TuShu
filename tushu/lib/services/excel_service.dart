import 'dart:io';
import 'package:excel/excel.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import '../models/template.dart';

class ExcelService {
  /// 创建新Excel文件
  static Future<String?> createNew({
    required Map<String, String> data,
    required TemplateType templateType,
  }) async {
    final excel = Excel.createExcel();
    final sheet = excel['Sheet1'];

    // 写入表头
    sheet.appendRow(data.keys.toList());

    // 写入数据行
    sheet.appendRow(data.values.toList());

    // 保存文件
    final fileName = await _getAvailableFileName(templateType);
    return await _saveExcel(excel, fileName);
  }

  /// 追加到已有Excel文件
  static Future<String?> appendToExisting({
    required Map<String, String> data,
    required TemplateType templateType,
  }) async {
    final fileName = _getFileName(templateType);
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$fileName');

    Excel excel;
    if (await file.exists()) {
      // 读取已有文件
      final bytes = await file.readAsBytes();
      excel = Excel.decodeBytes(bytes);
    } else {
      // 不存在则新建
      excel = Excel.createExcel();
    }

    final sheet = excel['Sheet1'];

    final headers = _readHeader(sheet);

    // 如果Sheet为空，先写表头；否则按既有表头顺序追加，避免列错位。
    if (headers.isEmpty) {
      final newHeaders = data.keys.toList();
      sheet.appendRow(newHeaders);
      sheet.appendRow(newHeaders.map((field) => data[field] ?? "").toList());
    } else {
      final extraFields = data.keys.where((field) => !headers.contains(field)).toList();
      if (extraFields.isNotEmpty) {
        throw Exception("追加失败，字段与已有表头不一致: ${extraFields.join(', ')}");
      }
      sheet.appendRow(headers.map((field) => data[field] ?? "").toList());
    }

    return await _saveExcel(excel, fileName);
  }

  /// 保存Excel到本地
  static Future<String?> _saveExcel(Excel excel, String fileName) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/$fileName');
      await file.writeAsBytes(excel.encode()!);
      return file.path;
    } catch (e) {
      return null;
    }
  }

  /// 生成文件名
  static String _getFileName(TemplateType type) {
    final now = DateTime.now();
    final dateStr = "${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}";
    switch (type) {
      case TemplateType.businessCard:
        return "名片汇总_$dateStr.xlsx";
      case TemplateType.invoice:
        return "发票汇总_$dateStr.xlsx";
      case TemplateType.custom:
        return "识别结果_$dateStr.xlsx";
    }
  }

  static Future<String> _getAvailableFileName(TemplateType type) async {
    final dir = await getApplicationDocumentsDirectory();
    final baseName = _getFileName(type);
    final dotIndex = baseName.lastIndexOf('.');
    final name = dotIndex == -1 ? baseName : baseName.substring(0, dotIndex);
    final extension = dotIndex == -1 ? '' : baseName.substring(dotIndex);

    var candidate = baseName;
    var suffix = 1;
    while (await File('${dir.path}/$candidate').exists()) {
      candidate = '${name}_$suffix$extension';
      suffix++;
    }
    return candidate;
  }

  static List<String> _readHeader(Sheet sheet) {
    if (sheet.maxRows == 0 || sheet.rows.isEmpty) {
      return [];
    }
    return sheet.rows.first
        .map((cell) => cell?.value?.toString().trim() ?? "")
        .where((value) => value.isNotEmpty)
        .toList();
  }

  /// 分享文件
  static Future<void> shareFile(String filePath) async {
    await Share.shareXFiles([XFile(filePath)]);
  }
}