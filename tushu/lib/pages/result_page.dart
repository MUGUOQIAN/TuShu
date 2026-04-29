import 'dart:io';
import 'package:flutter/material.dart';
import '../models/template.dart';
import '../services/excel_service.dart';

class ResultPage extends StatefulWidget {
  final String imagePath;
  final Map<String, String> extractedData;
  final TemplateType templateType;

  const ResultPage({
    super.key,
    required this.imagePath,
    required this.extractedData,
    required this.templateType,
  });

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  late Map<String, String> _editableData;
  bool _isAppending = false;

  @override
  void initState() {
    super.initState();
    _editableData = Map.from(widget.extractedData);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("识别结果"),
        actions: [
          // "追加模式"开关
          Row(
            children: [
              const Text("追加", style: TextStyle(fontSize: 14)),
              Switch(
                value: _isAppending,
                onChanged: (v) => setState(() => _isAppending = v),
              ),
            ],
          ),
        ],
      ),
      body: Column(
        children: [
          // 原图缩略图（点击可放大）
          SizedBox(
            height: 180,
            child: GestureDetector(
              onTap: () => _showFullImage(),
              child: Image.file(File(widget.imagePath), fit: BoxFit.cover),
            ),
          ),
          const Divider(),
          // 可编辑的表格
          Expanded(
            child: _editableData.isEmpty
                ? const Center(child: Text("未识别到数据"))
                : ListView(
                    children: _editableData.entries.map((entry) {
                      return ListTile(
                        title: Text(entry.key, style: const TextStyle(fontWeight: FontWeight.bold)),
                        subtitle: TextFormField(
                          initialValue: entry.value,
                          onChanged: (newValue) {
                            _editableData[entry.key] = newValue;
                          },
                          decoration: const InputDecoration(
                            border: InputBorder.none,
                            isDense: true,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
          ),
        ],
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            // 导出Excel
            Expanded(
              child: ElevatedButton.icon(
                icon: const Icon(Icons.table_chart),
                label: const Text("导出Excel"),
                onPressed: _exportToExcel,
              ),
            ),
            const SizedBox(width: 12),
            // 分享
            Expanded(
              child: OutlinedButton.icon(
                icon: const Icon(Icons.share),
                label: const Text("分享"),
                onPressed: _shareData,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 导出为Excel文件
  Future<void> _exportToExcel() async {
    try {
      String? filePath;
      if (_isAppending) {
        filePath = await ExcelService.appendToExisting(
          data: _editableData,
          templateType: widget.templateType,
        );
      } else {
        filePath = await ExcelService.createNew(
          data: _editableData,
          templateType: widget.templateType,
        );
      }

     if (filePath != null && mounted) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Text("已保存到: $filePath"),
      action: SnackBarAction(
        label: "分享",
        onPressed: () {
          // filePath 这里一定不为 null，但编译器不知道
          // 直接用非空断言
          ExcelService.shareFile(filePath!);
        },
      ),
    ),
  );
}
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("导出失败: $e")),
      );
    }
  }

  Future<void> _shareData() async {
    // 将数据转为CSV文本分享
    final buffer = StringBuffer();
    buffer.writeln("字段,值");
    _editableData.forEach((key, value) {
      buffer.writeln("$key,$value");
    });
    // 使用share_plus分享文本
    // Share.share(buffer.toString());
  }

  void _showFullImage() {
    showDialog(
      context: context,
      builder: (_) => Dialog(
        child: InteractiveViewer(
          child: Image.file(File(widget.imagePath)),
        ),
      ),
    );
  }
}