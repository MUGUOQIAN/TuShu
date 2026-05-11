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

    final existingHeaders = _readHeaderRow(sheet);
    if (existingHeaders.isEmpty) {
      final headers = data.keys.toList();
      sheet.appendRow(headers);
      sheet.appendRow(rowForHeaders(data, headers));
    } else {
      final headers = headersForAppend(existingHeaders, data.keys);
      if (headers.length != existingHeaders.length) {
        _writeHeaderRow(sheet, headers);
      }
      sheet.appendRow(rowForHeaders(data, headers));
    }

    return await _saveExcel(excel, fileName);
  }

  /// 保留已有表头顺序，并把本次新增字段追加到表头末尾。
  static List<String> headersForAppend(
    List<String> existingHeaders,
    Iterable<String> dataKeys,
  ) {
    final headers = List<String>.from(existingHeaders);
    for (final key in dataKeys) {
      if (!headers.contains(key)) {
        headers.add(key);
      }
    }
    return headers;
  }

  /// 按表头顺序生成数据行，避免Map插入顺序变化导致列错位。
  static List<String> rowForHeaders(
    Map<String, String> data,
    List<String> headers,
  ) {
    return headers.map((header) => data[header] ?? "").toList();
  }

  static List<String> _readHeaderRow(Sheet sheet) {
    if (sheet.maxRows == 0 || sheet.maxCols == 0 || sheet.rows.isEmpty) {
      return [];
    }

    return sheet.rows.first
        .map((cell) => cell?.value?.toString().trim() ?? "")
        .where((header) => header.isNotEmpty)
        .toList();
  }

  static void _writeHeaderRow(Sheet sheet, List<String> headers) {
    for (var i = 0; i < headers.length; i++) {
      sheet
          .cell(CellIndex.indexByColumnRow(columnIndex: i, rowIndex: 0))
          .value = headers[i];
    }
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

  /// 分享文件
  static Future<void> shareFile(String filePath) async {
    await Share.shareXFiles([XFile(filePath)]);
  }
}