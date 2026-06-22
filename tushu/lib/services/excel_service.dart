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
    final dir = await getApplicationDocumentsDirectory();
    final fileName = await _getAvailableFileName(dir, _getFileName(templateType));
    return await _saveExcel(excel, fileName, directory: dir);
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

    final headers = data.keys.toList();

    // 如果Sheet为空，先写表头
    if (sheet.maxCols == 0) {
      sheet.appendRow(headers);
    } else {
      final existingHeaders = _readHeader(sheet);
      if (!_sameHeaders(existingHeaders, headers)) {
        throw StateError("已有Excel表头与当前字段不一致，请新建导出");
      }
    }

    // 追加数据行
    sheet.appendRow(headers.map((header) => data[header] ?? "").toList());

    return await _saveExcel(excel, fileName);
  }

  /// 保存Excel到本地
  static Future<String?> _saveExcel(
    Excel excel,
    String fileName, {
    Directory? directory,
  }) async {
    try {
      final dir = directory ?? await getApplicationDocumentsDirectory();
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

  static Future<String> _getAvailableFileName(Directory dir, String fileName) async {
    final original = File('${dir.path}/$fileName');
    if (!await original.exists()) {
      return fileName;
    }

    final dotIndex = fileName.lastIndexOf('.');
    final baseName = dotIndex == -1 ? fileName : fileName.substring(0, dotIndex);
    final extension = dotIndex == -1 ? '' : fileName.substring(dotIndex);
    var index = 1;
    while (true) {
      final candidate = '${baseName}_$index$extension';
      if (!await File('${dir.path}/$candidate').exists()) {
        return candidate;
      }
      index++;
    }
  }

  static List<String> _readHeader(Sheet sheet) {
    if (sheet.rows.isEmpty) {
      return [];
    }
    return sheet.rows.first
        .map((cell) => cell?.value?.toString().trim() ?? "")
        .where((value) => value.isNotEmpty)
        .toList();
  }

  static bool _sameHeaders(List<String> left, List<String> right) {
    if (left.length != right.length) {
      return false;
    }
    for (var i = 0; i < left.length; i++) {
      if (left[i] != right[i]) {
        return false;
      }
    }
    return true;
  }

  /// 分享文件
  static Future<void> shareFile(String filePath) async {
    await Share.shareXFiles([XFile(filePath)]);
  }
}