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
    file_data_uri = _prepare_image_data_uri(image_base64)
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

    # 1) 仅把含目标业务字段的对象视为最终结果，避免把 layout 外壳误当结果。
    # 自定义模板字段名可能与 layout 元数据（如 model/id）碰撞，始终走文本二次结构化。
    if template_type != "custom":
        direct_result = _find_expected_field_dict(data, expected_fields)
        if direct_result is not None:
            return direct_result

    # 2) 从 layout_parsing 结果中提取文本，再按模板做字段抽取。
    text_chunks = _extract_text_chunks(data)
    if template_type == "business_card":
        return _map_business_card_fields(text_chunks)
    if not text_chunks:
        return {}
    return _structure_text_fields(text_chunks, prompt, expected_fields)


def _prepare_image_data_uri(
    image_base64: str,
    max_edge: int = 1280,
    max_bytes: int = 450 * 1024,
    max_pixels: int = 25_000_000,
) -> str:
    image_bytes = base64.b64decode(image_base64, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as img:
        original_format = (img.format or "").upper()
        width, height = img.size
        # 压缩率极高的图片可能字节很小但解码后占用数百 MB；必须在
        # convert/resize 触发完整像素分配前拒绝。
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError("图片像素尺寸过大")
        long_edge = max(width, height)
        mime_type = _mime_type_for_format(original_format)

        if mime_type and long_edge <= max_edge and len(image_bytes) <= max_bytes:
            return f"data:{mime_type};base64,{image_base64}"

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
                encoded = base64.b64encode(result).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
            quality -= 10

    return f"data:image/jpeg;base64,{image_base64}"


def _mime_type_for_format(image_format: str) -> str:
    mapping = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    return mapping.get(image_format, "")


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


def _find_expected_field_dict(data: dict, expected_fields: list[str]) -> dict | None:
    if not expected_fields:
        return None
    for candidate in _candidate_dicts(data):
        if any(field in candidate for field in expected_fields):
            return candidate
    return None


def _candidate_dicts(data: dict):
    stack = [data]
    seen_ids = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)
        yield current
        for key in ("data", "result"):
            nested = current.get(key)
            if isinstance(nested, dict):
                stack.append(nested)


def _structure_text_fields(
    text_chunks: list[str], prompt: str, expected_fields: list[str]
) -> dict:
    if not GLM_API_KEY:
        raise ValueError("缺少 GLM_API_KEY 环境变量")

    ocr_text = "\n".join(text_chunks)
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": STRUCTURE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你只根据用户提供的OCR文本抽取字段，并且只输出JSON对象。"
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"目标字段：{', '.join(expected_fields)}\n\n"
                    "OCR文本如下：\n"
                    f"{ocr_text}"
                )
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"GLM结构化请求失败 status={resp.status_code}, body={resp.text[:500]}")

    data = resp.json()
    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"GLM结构化响应缺少content: {data}") from exc

    parsed = _parse_json_from_response(content)
    nested = _find_expected_field_dict(parsed, expected_fields)
    return nested if nested is not None else parsed


def _extract_text_chunks(data: dict) -> list[str]:
    chunks: list[str] = []

    def append_text(value: str) -> None:
        for line in value.splitlines():
            line = line.strip()
            if line:
                chunks.append(line)

    def collect(value) -> None:
        if isinstance(value, str):
            append_text(value)
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            for key in ("content", "text", "md_results", "layout_details", "data", "result"):
                if key in value:
                    collect(value[key])

    collect(data)

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

    company_chunks: list[str] = []
    title_chunks: list[str] = []
    address_chunks: list[tuple[bool, str]] = []

    for chunk in chunks:
        is_company_chunk = any(k in chunk for k in company_keywords)
        if is_company_chunk:
            company_chunks.append(chunk)
        # 公司名常含“销售/顾问”，不能抢占独立职位行。
        if any(k in chunk for k in title_keywords) and not is_company_chunk:
            title_chunks.append(chunk)
        # 二字中文名（如“张路”）含路/街/号，不能抢占真实地址行。
        # 公司名常含“市/路/区/省”，优先使用非公司地址行。
        if any(k in chunk for k in address_keywords) and not _is_two_char_cn_name(chunk):
            address_chunks.append((is_company_chunk, chunk))

    company = company_chunks[0] if company_chunks else ""
    title = title_chunks[0] if title_chunks else ""
    dedicated_addresses = [c for is_company_chunk, c in address_chunks if not is_company_chunk]
    if dedicated_addresses:
        address = dedicated_addresses[0]
    elif address_chunks:
        # 仅有“公司名+地址”混排行时，保留该行以免地址全空。
        address = address_chunks[0][1]
    else:
        address = ""

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
        "电话",
        "手机",
        "邮箱",
        "@",
        "www",
        ".com",
        "factory",
    )
    # 路/街/号只跳过非二字中文名，避免“张路”被当成地址行丢掉。
    cn_address_markers = ("路", "街", "号")
    # 英文地址/联系词必须按整词匹配，否则 Addison/Broadway/Ismail
    # 会被 add/road/mail 子串误跳过。
    skip_address_words = ("add", "road", "district", "room", "building", "mail", "email")
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
        if any(m in c for m in cn_address_markers) and not _is_two_char_cn_name(c):
            continue
        if _has_whole_word(low, skip_address_words):
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
            looks_like_company = _has_whole_word(en_low, company_en_keywords)
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
        looks_like_company = _has_whole_word(en_low, company_en_keywords)
        if not _looks_like_address_phrase(en_name) and not looks_like_company:
            return en_name
    return ""


def _looks_like_address_phrase(value: str) -> bool:
    address_terms = ("factory", "add", "road", "district", "building", "room")
    return _has_whole_word(value.lower(), address_terms)


def _has_whole_word(value: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(keyword.lower())}\b", value) for keyword in keywords)


def _looks_like_valid_cn_name(value: str) -> bool:
    if len(value) < 2 or len(value) > 4:
        return False
    # 常见地名/公司前缀，避免把“上海玖协”当人名
    non_name_prefixes = ("上海", "北京", "广州", "深圳", "中国", "公司", "集团")
    if any(value.startswith(prefix) for prefix in non_name_prefixes):
        return False
    # 3 字及以上含路/街/号更像路名（中山路），二字名（张路）仍保留
    if len(value) >= 3 and any(marker in value for marker in ("路", "街", "号")):
        return False
    return True


def _is_two_char_cn_name(value: str) -> bool:
    text = value.strip()
    return bool(re.fullmatch(r"[\u4e00-\u9fa5]{2}", text)) and _looks_like_valid_cn_name(text)