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

    // 如果Sheet为空，先写表头
    final headers = _readHeader(sheet);
    if (headers.isEmpty) {
      final newHeaders = data.keys.toList();
      sheet.appendRow(newHeaders);
      sheet.appendRow(newHeaders.map((key) => data[key] ?? '').toList());
    } else {
      final existingFields = headers.toSet();
      final incomingFields = data.keys.toSet();
      if (existingFields.length != headers.length ||
          existingFields.length != incomingFields.length ||
          !existingFields.containsAll(incomingFields)) {
        throw Exception('字段与已有Excel不一致，无法追加');
      }
      // 必须按已有表头顺序写值；Map 插入顺序可能与文件列顺序不同。
      sheet.appendRow(headers.map((key) => data[key] ?? '').toList());
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
    var counter = 1;
    while (await File('${dir.path}/$candidate').exists()) {
      candidate = '${name}_$counter$extension';
      counter += 1;
    }
    return candidate;
  }

  static List<String> _readHeader(Sheet sheet) {
    if (sheet.maxRows == 0 || sheet.rows.isEmpty) {
      return const [];
    }
    return sheet.rows.first
        .map((cell) => cell?.value?.toString().trim() ?? '')
        .where((value) => value.isNotEmpty)
        .toList();
  }

  /// 分享文件
  static Future<void> shareFile(String filePath) async {
    await Share.shareXFiles([XFile(filePath)]);
  }
}