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
    var headers = _readHeader(sheet);

    // 如果Sheet为空，先写表头
    if (headers.isEmpty) {
      headers = data.keys.toList();
      sheet.appendRow(headers);
    } else {
      final existing = headers.toSet();
      final incoming = data.keys.toSet();
      if (existing.length != incoming.length || !existing.containsAll(incoming)) {
        throw StateError("现有Excel表头与当前字段不一致，无法安全追加");
      }
    }

    // 按既有表头顺序追加，避免 Map 顺序变化导致错列。
    sheet.appendRow(headers.map((field) => data[field] ?? "").toList());

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
      final file = overwrite
          ? File('${dir.path}/$fileName')
          : await _uniqueFile(dir, fileName);
      await file.writeAsBytes(excel.encode()!, flush: true);
      return file.path;
    } catch (e) {
      return null;
    }
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

  static Future<File> _uniqueFile(Directory dir, String fileName) async {
    final dot = fileName.lastIndexOf('.');
    final base = dot == -1 ? fileName : fileName.substring(0, dot);
    final ext = dot == -1 ? "" : fileName.substring(dot);
    var candidate = File('${dir.path}/$fileName');
    var index = 1;
    while (await candidate.exists()) {
      candidate = File('${dir.path}/${base}_$index$ext');
      index += 1;
    }
    return candidate;
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