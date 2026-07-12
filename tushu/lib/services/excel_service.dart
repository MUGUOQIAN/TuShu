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
    final headers = data.keys.toList();
    sheet.appendRow(headers);

    // 写入数据行
    sheet.appendRow(headers.map((header) => data[header] ?? '').toList());

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

    // 如果Sheet为空，先写表头
    var headers = _readHeaders(sheet);
    if (headers.isEmpty) {
      headers = data.keys.toList();
      sheet.appendRow(headers);
    } else {
      final missing = headers.where((header) => !data.containsKey(header)).toList();
      final extra = data.keys.where((header) => !headers.contains(header)).toList();
      if (missing.isNotEmpty || extra.isNotEmpty) {
        throw StateError("追加失败：字段与已有Excel表头不一致");
      }
    }

    // 追加数据行
    sheet.appendRow(headers.map((header) => data[header] ?? '').toList());

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
          : await _nextAvailableFile(dir, fileName);
      await file.writeAsBytes(excel.encode()!);
      return file.path;
    } catch (e) {
      return null;
    }
  }

  static List<String> _readHeaders(Sheet sheet) {
    if (sheet.maxRows == 0 || sheet.maxCols == 0) {
      return [];
    }
    return List.generate(sheet.maxCols, (columnIndex) {
      final cell = sheet.cell(CellIndex.indexByColumnRow(
        columnIndex: columnIndex,
        rowIndex: 0,
      ));
      return cell.value?.toString().trim() ?? '';
    }).where((value) => value.isNotEmpty).toList();
  }

  static Future<File> _nextAvailableFile(Directory dir, String fileName) async {
    final dotIndex = fileName.lastIndexOf('.');
    final baseName = dotIndex == -1 ? fileName : fileName.substring(0, dotIndex);
    final extension = dotIndex == -1 ? '' : fileName.substring(dotIndex);

    var candidate = File('${dir.path}/$fileName');
    var suffix = 1;
    while (await candidate.exists()) {
      candidate = File('${dir.path}/${baseName}_$suffix$extension');
      suffix++;
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