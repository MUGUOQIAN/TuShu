import re
from typing import List, Dict


def validate_and_clean(result: dict, expected_fields: List[str]) -> dict:
    """
    校验并清洗大模型返回的结果。
    1. 补全缺失字段（设为空字符串）
    2. 清理金额格式
    3. 确保日期格式统一
    """
    if not isinstance(result, dict):
        raise TypeError(f"validate_and_clean期望dict，实际为: {type(result).__name__}")

    cleaned = {}

    for field in expected_fields:
        value = result.get(field, "")
        if value is None:
            value = ""

        # 清洗金额字段：移除¥、逗号、空格，但保留数字和小数点
        if any(keyword in field for keyword in ["金额", "价税合计", "税额"]):
            value = _clean_amount(str(value))

        # 清洗日期字段：尝试统一格式
        if "日期" in field or "时间" in field:
            value = _normalize_date(str(value))

        cleaned[field] = str(value).strip()

    return cleaned


def _clean_amount(value: str) -> str:
    """提取金额中的数字和小数点"""
    if not value:
        return ""
    # 移除货币符号、逗号、空格
    cleaned = re.sub(r'[¥$,，、\s]', '', value)
    # 确保只保留合法的数字格式
    match = re.search(r'\d+\.?\d*', cleaned)
    return match.group() if match else ""


def _normalize_date(value: str) -> str:
    """
    尝试将各种日期格式转为 YYYY-MM-DD
    此处简化处理，实际可用dateutil库更精准
    """
    # 匹配常见的2024年1月1日、2024-01-01、2024/01/01等
    patterns = [
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            y, m, d = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return value