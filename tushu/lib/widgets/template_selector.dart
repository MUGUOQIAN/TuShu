import 'package:flutter/material.dart';
import '../models/template.dart';

class TemplateSelector extends StatelessWidget {
  final TemplateType selected;
  final String customFields;
  final Function(TemplateType, String) onTemplateChanged;

  const TemplateSelector({
    super.key,
    required this.selected,
    required this.customFields,
    required this.onTemplateChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 模板快捷选择
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: TemplateType.values.map((t) {
                final isSelected = t == selected;
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: ChoiceChip(
                    label: Text(t.label),
                    selected: isSelected,
                    onSelected: (_) {
                      if (t == TemplateType.custom) {
                        _showCustomDialog(context);
                      } else {
                        onTemplateChanged(t, "");
                      }
                    },
                    selectedColor: Colors.blue,
                    labelStyle: TextStyle(
                      color: isSelected ? Colors.white : Colors.white70,
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
          // 自定义字段展示
          if (selected == TemplateType.custom && customFields.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                "自定义字段: $customFields",
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
            ),
        ],
      ),
    );
  }

  void _showCustomDialog(BuildContext context) {
    final controller = TextEditingController(text: customFields);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("输入要提取的字段"),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: "例如：姓名, 电话, 地址",
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("取消"),
          ),
          TextButton(
            onPressed: () {
              onTemplateChanged(TemplateType.custom, controller.text);
              Navigator.pop(ctx);
            },
            child: const Text("确定"),
          ),
        ],
      ),
    );
  }
}