import base64
import io
import json
import re
from typing import Optional, Sequence
import requests
from PIL import Image
from config import GLM_API_KEY, GLM_TEXT_MODEL, PRIMARY_MODEL

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow>=9.1
except AttributeError:
    _LANCZOS = Image.LANCZOS  # Pillow<9.1


def call_llm(
    image_base64: str,
    prompt: str,
    model: str = PRIMARY_MODEL,
    expected_fields: Optional[Sequence[str]] = None,
) -> dict:
    """
    调用大模型API（仅GLM），返回解析后的JSON结果。
    """
    return _call_model(image_base64, prompt, model, expected_fields)


def _call_model(
    image_base64: str,
    prompt: str,
    model: str,
    expected_fields: Optional[Sequence[str]],
) -> dict:
    """实际调用模型API"""
    if model == "glm-ocr":
        return _call_glm(image_base64, prompt, expected_fields)
    else:
        raise ValueError(f"不支持的模型: {model}")


def _call_glm(
    image_base64: str,
    prompt: str,
    expected_fields: Optional[Sequence[str]] = None,
) -> dict:
    """调用智谱 GLM OCR API（layout_parsing）"""
    if not GLM_API_KEY:
        raise ValueError("缺少 GLM_API_KEY 环境变量")

    url = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    # GLM-OCR 要求 file 使用 URL 或 data URI；先压缩可显著降低上传超时概率。
    compressed_base64 = _compress_base64_image(image_base64)
    file_data_uri = f"data:image/jpeg;base64,{compressed_base64}"
    payload = {
        "model": "glm-ocr",
        "file": file_data_uri,
        "need_layout_visualization": False,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"GLM请求失败 status={resp.status_code}, body={resp.text[:500]}")
    data = resp.json()
    if not isinstance(data, dict):
        return {}

    # 1) 仅当返回内容确实包含目标业务字段时，才当作结构化 JSON。
    direct_result = _extract_direct_result(data, expected_fields)
    if direct_result is not None:
        return direct_result

    # 2) 从 layout_parsing 结果中提取文本。名片保留本地启发式；其他模板需二次结构化。
    text_chunks = _extract_text_chunks(data)
    if not text_chunks:
        raise RuntimeError("GLM OCR 未返回有效文本")
    if _is_business_card_fields(expected_fields):
        return _map_business_card_fields(text_chunks)
    return _extract_structured_fields_from_text(text_chunks, prompt, expected_fields)


def _compress_base64_image(
    image_base64: str, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> str:
    image_bytes = base64.b64decode(image_base64)
    # 输入已较小则不再二次压缩，避免姓名等细节文字丢失。
    if len(image_bytes) <= 300 * 1024:
        return image_base64

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        width, height = img.size
        long_edge = max(width, height)
        if long_edge > max_edge:
            scale = max_edge / long_edge
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, _LANCZOS)

        quality = 75
        while quality >= 35:
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            result = output.getvalue()
            if len(result) <= max_bytes or quality == 35:
                return base64.b64encode(result).decode("utf-8")
            quality -= 10

    return image_base64


def _parse_json_from_response(content: str) -> dict:
    """
    从模型返回的文本中提取JSON对象。
    处理模型偶尔输出的"```json ... ```"格式。
    """
    content = content.strip()
    # 移除可能的Markdown代码块标记
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    # 尝试找JSON部分
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        json_str = content[start:end]
    else:
        json_str = content

    return json.loads(json_str)


def _extract_direct_result(
    data: dict,
    expected_fields: Optional[Sequence[str]],
) -> Optional[dict]:
    for candidate in (data, data.get("data"), data.get("result")):
        if isinstance(candidate, dict) and _contains_expected_field(candidate, expected_fields):
            return candidate
    return None


def _contains_expected_field(
    candidate: dict,
    expected_fields: Optional[Sequence[str]],
) -> bool:
    if not expected_fields:
        return False
    return any(field in candidate for field in expected_fields)


def _is_business_card_fields(expected_fields: Optional[Sequence[str]]) -> bool:
    return set(expected_fields or []) == {"姓名", "公司", "职位", "手机", "座机", "邮箱", "地址"}


def _extract_structured_fields_from_text(
    chunks: list[str],
    prompt: str,
    expected_fields: Optional[Sequence[str]],
) -> dict:
    if not expected_fields:
        raise RuntimeError("缺少目标字段，无法结构化 OCR 文本")

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    ocr_text = "\n".join(chunks)
    user_prompt = (
        f"{prompt.strip()}\n\n"
        "以下是 OCR 从图片中识别出的原始文本，请只基于这些文本抽取字段：\n"
        f"{ocr_text}"
    )
    payload = {
        "model": GLM_TEXT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是严谨的信息抽取助手。只输出一个 JSON 对象，不要输出解释文字。",
            },
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"GLM结构化抽取失败 status={resp.status_code}, body={resp.text[:500]}")
    data = resp.json()
    content = _extract_chat_content(data)
    parsed = _parse_json_from_response(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("GLM结构化抽取结果不是JSON对象")
    if not _contains_expected_field(parsed, expected_fields):
        raise RuntimeError("GLM结构化抽取未返回目标字段")
    return parsed


def _extract_chat_content(data: dict) -> str:
    if not isinstance(data, dict):
        raise RuntimeError(f"GLM结构化抽取返回类型异常: {type(data).__name__}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("GLM结构化抽取返回缺少choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("GLM结构化抽取choices格式异常")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("GLM结构化抽取返回缺少message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("GLM结构化抽取返回内容为空")
    return content


def _extract_text_chunks(data: dict) -> list[str]:
    chunks: list[str] = []

    for source in _layout_sources(data):
        md_results = source.get("md_results")
        if isinstance(md_results, str) and md_results.strip():
            chunks.append(md_results.strip())
        elif isinstance(md_results, list):
            for item in md_results:
                if isinstance(item, str) and item.strip():
                    chunks.append(item.strip())

        _collect_layout_content(source.get("layout_details"), chunks)

    # 去重保序
    seen = set()
    ordered = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _layout_sources(data: dict) -> list[dict]:
    sources = [data]
    for key in ("data", "result"):
        nested = data.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    return sources


def _collect_layout_content(value, chunks: list[str]):
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content.strip())
        for child in value.values():
            _collect_layout_content(child, chunks)
    elif isinstance(value, list):
        for item in value:
            _collect_layout_content(item, chunks)


def _map_business_card_fields(chunks: list[str]) -> dict:
    text = "\n".join(chunks)

    email = _first_match(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    # 手机优先匹配中国大陆11位
    mobile = _first_match(r"(?<!\d)1[3-9]\d{9}(?!\d)", text)

    # 座机示例：021-12345678 / 02112345678
    landline = _first_match(r"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)", text)
    if landline == mobile:
        landline = ""

    company = ""
    name = ""
    title = ""
    address = ""

    company_keywords = ("公司", "有限", "集团", "Co.,Ltd", "Co., Ltd", "Ltd", "Inc", "Corporation")
    title_keywords = ("经理", "总监", "主管", "工程师", "销售", "总裁", "主任", "顾问", "Designer", "Manager")
    address_keywords = ("地址", "路", "街", "号", "区", "市", "省", "Address")

    for chunk in chunks:
        if not company and any(k in chunk for k in company_keywords):
            company = chunk
        if not title and any(k in chunk for k in title_keywords):
            title = chunk
        if not address and any(k in chunk for k in address_keywords):
            address = chunk

    name = _extract_name(chunks)

    return {
        "姓名": name or "",
        "公司": company or "",
        "职位": title or "",
        "手机": mobile or "",
        "座机": landline or "",
        "邮箱": email or "",
        "地址": address or "",
    }


def _first_match(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m and m.lastindex else (m.group(0) if m else "")


def _extract_name(chunks: list[str]) -> str:
    text = "\n".join(chunks)

    # 先在单行里找疑似姓名，避免全局命中公司简称（如“上海玖协”）
    skip_keywords = (
        "公司",
        "有限",
        "集团",
        "地址",
        "路",
        "街",
        "号",
        "电话",
        "手机",
        "邮箱",
        "mail",
        "@",
        "www",
        ".com",
        "factory",
        "add",
        "road",
        "district",
        "room",
        "building",
    )
    title_keywords = ("经理", "总监", "主管", "工程师", "销售", "总裁", "主任", "顾问", "Manager", "Director")
    company_en_keywords = ("co", "ltd", "inc", "corporation", "machinery", "shanghai", "jiuxie", "company")
    cn_name_pattern = re.compile(r"[\u4e00-\u9fa5]{2,4}")
    en_name_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
    cn_en_combo_pattern = re.compile(
        r"([\u4e00-\u9fa5]{2,4})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
    )

    english_name_candidate = ""

    for i, chunk in enumerate(chunks):
        c = chunk.strip()
        if not c:
            continue
        low = c.lower()
        if any(k.lower() in low for k in skip_keywords):
            continue
        if any(k in c for k in title_keywords):
            continue

        # 优先中英同现的人名行，例如：赵美娜 Shermin Zhao
        combo = cn_en_combo_pattern.search(c)
        if combo:
            cn_name = combo.group(1)
            en_name = combo.group(2)
            if _looks_like_valid_cn_name(cn_name) and not _looks_like_address_phrase(en_name):
                return f"{cn_name} ({en_name})"

        cn = cn_name_pattern.search(c)
        if cn:
            cn_name = cn.group(0)
            if _looks_like_valid_cn_name(cn_name):
                return cn_name
        en = en_name_pattern.search(c)
        if en:
            en_name = en.group(0)
            en_low = en_name.lower()
            looks_like_company = any(k in en_low for k in company_en_keywords)
            if not _looks_like_address_phrase(en_name) and not looks_like_company:
                # 若姓名行邻近职位行，优先作为最终姓名。
                prev_chunk = chunks[i - 1] if i > 0 else ""
                next_chunk = chunks[i + 1] if i + 1 < len(chunks) else ""
                near_title = any(k in prev_chunk for k in title_keywords) or any(k in next_chunk for k in title_keywords)
                if near_title:
                    return en_name
                if not english_name_candidate:
                    english_name_candidate = en_name

    # 回退：全局搜索
    if english_name_candidate:
        return english_name_candidate

    cn = cn_name_pattern.search(text)
    if cn:
        cn_name = cn.group(0)
        if _looks_like_valid_cn_name(cn_name):
            return cn_name
    en = en_name_pattern.search(text)
    if en:
        en_name = en.group(0)
        en_low = en_name.lower()
        looks_like_company = any(k in en_low for k in company_en_keywords)
        if not _looks_like_address_phrase(en_name) and not looks_like_company:
            return en_name
    return ""


def _looks_like_address_phrase(value: str) -> bool:
    low = value.lower()
    address_terms = ("factory", "add", "road", "district", "building", "room")
    return any(term in low for term in address_terms)


def _looks_like_valid_cn_name(value: str) -> bool:
    if len(value) < 2 or len(value) > 4:
        return False
    # 常见地名/公司前缀，避免把“上海玖协”当人名
    non_name_prefixes = ("上海", "北京", "广州", "深圳", "中国", "公司", "集团")
    return not any(value.startswith(prefix) for prefix in non_name_prefixes)