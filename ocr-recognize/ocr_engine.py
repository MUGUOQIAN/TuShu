import base64
import io
import json
import re
import requests
from PIL import Image
from typing import Optional
from config import GLM_API_KEY, PRIMARY_MODEL, STRUCTURE_MODEL

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow>=9.1
except AttributeError:
    _LANCZOS = Image.LANCZOS  # Pillow<9.1


def call_llm(
    image_base64: str,
    prompt: str,
    model: str = PRIMARY_MODEL,
    expected_fields: Optional[list[str]] = None,
    template_type: Optional[str] = None,
) -> dict:
    """
    调用大模型API（仅GLM），返回解析后的JSON结果。
    """
    return _call_model(image_base64, prompt, model, expected_fields, template_type)


def _call_model(
    image_base64: str,
    prompt: str,
    model: str,
    expected_fields: Optional[list[str]],
    template_type: Optional[str],
) -> dict:
    """实际调用模型API"""
    if model == "glm-ocr":
        return _call_glm(image_base64, prompt, expected_fields, template_type)
    else:
        raise ValueError(f"不支持的模型: {model}")


def _call_glm(
    image_base64: str,
    prompt: str,
    expected_fields: Optional[list[str]] = None,
    template_type: Optional[str] = None,
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
    file_data_uri = _build_glm_file_data_uri(image_base64)
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

    # 1) 只有包含业务字段的 JSON 才能直接返回；layout 外壳不是最终抽取结果。
    direct_result = _find_business_result(data, expected_fields)
    if direct_result is not None:
        return direct_result

    # 2) 从 layout_parsing 结果中提取文本。
    text_chunks = _extract_text_chunks(data)
    if template_type == "business_card" or not expected_fields:
        return _map_business_card_fields(text_chunks)

    # 3) 发票/自定义模板需要按用户字段再做一次结构化抽取。
    return _structure_text_fields(text_chunks, prompt, expected_fields)


def _build_glm_file_data_uri(image_base64: str) -> str:
    image_bytes = base64.b64decode(image_base64, validate=True)

    with Image.open(io.BytesIO(image_bytes)) as img:
        source_format = img.format or "JPEG"
        width, height = img.size
        long_edge = max(width, height)

        # 小图且尺寸已合适时保留原编码和 MIME，避免把 PNG/WebP 伪装成 JPEG。
        if len(image_bytes) <= 300 * 1024 and long_edge <= 1280:
            mime = Image.MIME.get(source_format, "image/jpeg")
            return f"data:{mime};base64,{image_base64}"

        compressed_base64 = _compress_image_bytes(img)
        return f"data:image/jpeg;base64,{compressed_base64}"


def _compress_base64_image(
    image_base64: str, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> str:
    image_bytes = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(image_bytes)) as img:
        return _compress_image_bytes(img, max_edge=max_edge, max_bytes=max_bytes)


def _compress_image_bytes(
    img: Image.Image, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> str:
    img = img.convert("RGB")
    width, height = img.size
    long_edge = max(width, height)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, _LANCZOS)

    quality = 75
    while quality >= 35:
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        result = output.getvalue()
        if len(result) <= max_bytes or quality == 35:
            return base64.b64encode(result).decode("utf-8")
        quality -= 10

    raise RuntimeError("图片压缩失败")


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


def _find_business_result(
    data: dict, expected_fields: Optional[list[str]] = None
) -> Optional[dict]:
    for candidate in _candidate_dicts(data):
        if _contains_expected_field(candidate, expected_fields):
            return candidate
    return None


def _candidate_dicts(data: dict) -> list[dict]:
    candidates: list[dict] = []
    if isinstance(data, dict):
        candidates.append(data)
        for key in ("data", "result"):
            value = data.get(key)
            if isinstance(value, dict):
                candidates.extend(_candidate_dicts(value))
    return candidates


def _contains_expected_field(
    candidate: dict, expected_fields: Optional[list[str]] = None
) -> bool:
    if not expected_fields:
        layout_keys = {"md_results", "layout_details", "pages"}
        return not layout_keys.intersection(candidate.keys())
    return any(field in candidate for field in expected_fields)


def _structure_text_fields(
    text_chunks: list[str], prompt: str, expected_fields: list[str]
) -> dict:
    if not text_chunks:
        return {field: "" for field in expected_fields}

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }
    ocr_text = "\n".join(text_chunks)
    structure_prompt = (
        f"{prompt.strip()}\n\n"
        "以下是 OCR 已识别出的文本，请只根据这些文本提取字段并输出 JSON：\n"
        f"{ocr_text}"
    )
    payload = {
        "model": STRUCTURE_MODEL,
        "messages": [{"role": "user", "content": structure_prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"GLM结构化请求失败 status={resp.status_code}, body={resp.text[:500]}")

    data = resp.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = _parse_json_from_response(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"结构化模型返回JSON不是对象类型: {type(parsed).__name__}")
    return parsed


def _extract_text_chunks(data: dict) -> list[str]:
    chunks: list[str] = []

    def add_text(value: str) -> None:
        for line in value.splitlines():
            line = line.strip()
            if line:
                chunks.append(line)

    def visit(node) -> None:
        if isinstance(node, dict):
            md_results = node.get("md_results")
            if isinstance(md_results, str):
                add_text(md_results)
            elif isinstance(md_results, list):
                for item in md_results:
                    if isinstance(item, str):
                        add_text(item)
                    else:
                        visit(item)

            content = node.get("content")
            if isinstance(content, str):
                add_text(content)

            for key in ("layout_details", "data", "result"):
                if key in node:
                    visit(node[key])
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(data)

    # 去重保序
    seen = set()
    ordered = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


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