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

    final fileName = _getFileName(templateType);
    return await _saveExcel(excel, fileName, overwrite: false);
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

    final headers = _headersFromSheet(sheet);
    if (headers.isEmpty) {
      final newHeaders = data.keys.toList();
      sheet.appendRow(newHeaders);
      sheet.appendRow(newHeaders.map((field) => data[field] ?? "").toList());
    } else {
      final headerSet = headers.toSet();
      final dataSet = data.keys.toSet();
      if (headerSet.length != dataSet.length || !headerSet.containsAll(dataSet)) {
        throw StateError("已有Excel表头与当前字段不一致，请关闭追加模式新建文件");
      }
      sheet.appendRow(headers.map((field) => data[field] ?? "").toList());
    }

    return await _saveExcel(excel, fileName);
  }

  /// 保存Excel到本地
  static Future<String?> _saveExcel(
    Excel excel,
    String fileName, {
    bool overwrite = true,
  }) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final targetName = overwrite ? fileName : await _availableFileName(dir, fileName);
      final file = File('${dir.path}/$targetName');
      await file.writeAsBytes(excel.encode()!);
      return file.path;
    } catch (e) {
      return null;
    }
  }

  static List<String> _headersFromSheet(Sheet sheet) {
    if (sheet.maxRows == 0 || sheet.rows.isEmpty) {
      return [];
    }
    return sheet.rows.first
        .map((cell) => cell?.value?.toString().trim() ?? "")
        .where((value) => value.isNotEmpty)
        .toList();
  }

  static Future<String> _availableFileName(Directory dir, String fileName) async {
    final file = File('${dir.path}/$fileName');
    if (!await file.exists()) {
      return fileName;
    }

    final dotIndex = fileName.lastIndexOf(".");
    final name = dotIndex == -1 ? fileName : fileName.substring(0, dotIndex);
    final extension = dotIndex == -1 ? "" : fileName.substring(dotIndex);
    var suffix = 1;
    while (true) {
      final candidate = "${name}_$suffix$extension";
      if (!await File('${dir.path}/$candidate').exists()) {
        return candidate;
      }
      suffix++;
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

  /// 分享文件
  static Future<void> shareFile(String filePath) async {
    await Share.shareXFiles([XFile(filePath)]);
  }
}