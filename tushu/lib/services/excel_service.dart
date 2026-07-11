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

    // 新建导出不能覆盖同一天已经保存的文件。
    final fileName = await _getUniqueFileName(templateType);
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

    // 如果Sheet为空，先写表头
    if (sheet.maxCols == 0) {
      sheet.appendRow(data.keys.toList());
      sheet.appendRow(data.values.toList());
      return await _saveExcel(excel, fileName);
    }

    final headers = _readHeaderRow(sheet);
    final newFields = data.keys.where((key) => !headers.contains(key)).toList();
    if (newFields.isNotEmpty) {
      throw StateError("当前文件缺少字段: ${newFields.join(', ')}，请新建文件导出");
    }

    // 按已有表头顺序写入，避免自定义字段变化时列错位。
    sheet.appendRow(headers.map((key) => data[key] ?? "").toList());

    return await _saveExcel(excel, fileName);
  }

  static List<String> _readHeaderRow(Sheet sheet) {
    final headers = <String>[];
    for (var col = 0; col < sheet.maxCols; col++) {
      final value = sheet
          .cell(CellIndex.indexByColumnRow(columnIndex: col, rowIndex: 0))
          .value;
      final text = value?.toString().trim() ?? "";
      if (text.isNotEmpty) {
        headers.add(text);
      }
    }
    return headers;
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

  static Future<String> _getUniqueFileName(TemplateType type) async {
    final dir = await getApplicationDocumentsDirectory();
    final baseName = _getFileName(type);
    final dotIndex = baseName.lastIndexOf('.');
    final name = dotIndex == -1 ? baseName : baseName.substring(0, dotIndex);
    final ext = dotIndex == -1 ? "" : baseName.substring(dotIndex);

    var candidate = baseName;
    var counter = 1;
    while (await File('${dir.path}/$candidate').exists()) {
      candidate = "${name}_$counter$ext";
      counter++;
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