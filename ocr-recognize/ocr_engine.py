import base64
import io
import json
import re
import requests
from PIL import Image
from config import GLM_API_KEY, PRIMARY_MODEL, STRUCTURE_MODEL

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow>=9.1
except AttributeError:
    _LANCZOS = Image.LANCZOS  # Pillow<9.1


def call_llm(
    image_base64: str,
    prompt: str,
    model: str = PRIMARY_MODEL,
    expected_fields: list[str] | None = None,
    template_type: str = "business_card",
) -> dict:
    """
    调用大模型API（仅GLM），返回解析后的JSON结果。
    """
    return _call_model(image_base64, prompt, model, expected_fields or [], template_type)


def _call_model(
    image_base64: str,
    prompt: str,
    model: str,
    expected_fields: list[str],
    template_type: str,
) -> dict:
    """实际调用模型API"""
    if model == "glm-ocr":
        return _call_glm(image_base64, prompt, expected_fields, template_type)
    else:
        raise ValueError(f"不支持的模型: {model}")


def _call_glm(
    image_base64: str,
    prompt: str,
    expected_fields: list[str],
    template_type: str,
) -> dict:
    """调用智谱 GLM OCR API（layout_parsing）"""
    if not GLM_API_KEY:
        raise ValueError("缺少 GLM_API_KEY 环境变量")

    url = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    # GLM-OCR 要求 file 使用 URL 或 data URI；必要时压缩可显著降低上传超时概率。
    file_data_uri = _image_data_uri_from_base64(image_base64)
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

    # 1) 只有真正包含业务字段的 JSON 才能跳过 layout 文本处理。
    structured = _find_structured_candidate(data, expected_fields)
    if structured is not None:
        return structured

    # 2) 从 layout_parsing 结果中提取 OCR 文本。
    text_chunks = _extract_text_chunks(data)
    if template_type == "business_card":
        return _map_business_card_fields(text_chunks)

    # 3) 发票/自定义字段需要将 OCR 文本再交给文本模型结构化。
    return _structure_text_fields(text_chunks, prompt, expected_fields)


def _compress_base64_image(
    image_base64: str, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> str:
    return _image_data_uri_from_base64(image_base64, max_edge, max_bytes).split(",", 1)[1]


def _image_data_uri_from_base64(
    image_base64: str, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> str:
    image_bytes = base64.b64decode(image_base64)

    with Image.open(io.BytesIO(image_bytes)) as img:
        original_format = (img.format or "JPEG").lower()
        width, height = img.size
        long_edge = max(width, height)

        # 输入已较小且分辨率合适则保留原格式，避免 PNG/WebP 被错误标成 JPEG。
        if len(image_bytes) <= 300 * 1024 and long_edge <= max_edge:
            mime_format = "jpeg" if original_format in ("jpg", "jpeg") else original_format
            if mime_format in ("jpeg", "png", "webp"):
                return f"data:image/{mime_format};base64,{image_base64}"

        img = img.convert("RGB")
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
                compressed_base64 = base64.b64encode(result).decode("utf-8")
                return f"data:image/jpeg;base64,{compressed_base64}"
            quality -= 10

    return f"data:image/jpeg;base64,{image_base64}"


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


def _extract_text_chunks(data: dict) -> list[str]:
    chunks: list[str] = []

    def add_text(value: str):
        for line in value.splitlines():
            stripped = line.strip()
            if stripped:
                chunks.append(stripped)

    def walk(node):
        if isinstance(node, str):
            add_text(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(data)

    # 去重保序
    seen = set()
    ordered = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _find_structured_candidate(data: dict, expected_fields: list[str]) -> dict | None:
    candidates: list[dict] = []

    def collect(node):
        if not isinstance(node, dict):
            return
        candidates.append(node)
        for key in ("data", "result"):
            value = node.get(key)
            if isinstance(value, dict):
                collect(value)

    collect(data)

    if expected_fields:
        for candidate in candidates:
            if any(field in candidate for field in expected_fields):
                return candidate
        return None

    for key in ("data", "result"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


def _structure_text_fields(chunks: list[str], prompt: str, expected_fields: list[str]) -> dict:
    if not chunks:
        return {}
    if not GLM_API_KEY:
        raise ValueError("缺少 GLM_API_KEY 环境变量")

    text = "\n".join(chunks)
    fields_text = ", ".join(expected_fields) if expected_fields else "所需字段"
    user_prompt = (
        f"{prompt}\n\n"
        "以下是 GLM-OCR layout_parsing 识别到的文本，请只基于这些文本抽取字段。\n"
        f"字段列表：{fields_text}\n\n"
        f"OCR文本：\n{text}"
    )
    payload = {
        "model": STRUCTURE_MODEL,
        "messages": [
            {"role": "system", "content": "你是信息抽取助手，只输出一个 JSON 对象，不要解释。"},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"GLM结构化请求失败 status={resp.status_code}, body={resp.text[:500]}")

    data = resp.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = _parse_json_from_response(content)
    structured = _find_structured_candidate(parsed, expected_fields) or parsed
    if expected_fields:
        return {field: structured.get(field, "") for field in expected_fields}
    return structured


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
    company_en_keywords = ("ltd", "inc", "corporation", "machinery", "shanghai", "jiuxie", "company")
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
            looks_like_company = _looks_like_company_en(en_low, company_en_keywords)
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
        looks_like_company = _looks_like_company_en(en_low, company_en_keywords)
        if not _looks_like_address_phrase(en_name) and not looks_like_company:
            return en_name
    return ""


def _looks_like_address_phrase(value: str) -> bool:
    low = value.lower()
    address_terms = ("factory", "add", "road", "district", "building", "room")
    return any(term in low for term in address_terms)


def _looks_like_company_en(value: str, keywords: tuple[str, ...]) -> bool:
    if any(keyword in value for keyword in keywords):
        return True
    return re.search(r"\bco\b", value) is not None


def _looks_like_valid_cn_name(value: str) -> bool:
    if len(value) < 2 or len(value) > 4:
        return False
    # 常见地名/公司前缀，避免把“上海玖协”当人名
    non_name_prefixes = ("上海", "北京", "广州", "深圳", "中国", "公司", "集团")
    return not any(value.startswith(prefix) for prefix in non_name_prefixes)