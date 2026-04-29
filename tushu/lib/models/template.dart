enum TemplateType {
  businessCard,
  invoice,
  custom;

  String get label {
    switch (this) {
      case TemplateType.businessCard:
        return "💼 名片";
      case TemplateType.invoice:
        return "🧾 发票";
      case TemplateType.custom:
        return "✍️ 自定义";
    }
  }
}