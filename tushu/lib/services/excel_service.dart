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

    var headers = _readHeaderRow(sheet);
    if (headers.isEmpty) {
      headers = data.keys.toList();
      sheet.appendRow(headers);
    } else {
      final incompatibleFields =
          data.keys.where((key) => !headers.contains(key)).toList();
      if (incompatibleFields.isNotEmpty) {
        throw StateError("字段与现有Excel表头不一致: ${incompatibleFields.join(', ')}");
      }
    }

    // 按已有表头顺序写入，避免字段顺序变化导致列错位。
    sheet.appendRow(headers.map((header) => data[header] ?? "").toList());

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
          : await _uniqueFile(dir.path, fileName);
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

  static List<String> _readHeaderRow(Sheet sheet) {
    if (sheet.maxRows == 0 || sheet.maxCols == 0) {
      return [];
    }
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

  static Future<File> _uniqueFile(String dirPath, String fileName) async {
    var candidate = File('$dirPath/$fileName');
    if (!await candidate.exists()) {
      return candidate;
    }

    final dot = fileName.lastIndexOf('.');
    final base = dot == -1 ? fileName : fileName.substring(0, dot);
    final extension = dot == -1 ? "" : fileName.substring(dot);
    var counter = 1;
    do {
      candidate = File('$dirPath/${base}_$counter$extension');
      counter += 1;
    } while (await candidate.exists());
    return candidate;
  }

  /// 分享文件
  static Future<void> shareFile(String filePath) async {
    await Share.shareXFiles([XFile(filePath)]);
  }
}