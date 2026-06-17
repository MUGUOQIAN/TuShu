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


class InvalidImageError(ValueError):
    """Raised when the request image cannot be decoded safely."""


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
    return _call_model(image_base64, prompt, model, expected_fields, template_type)


def _call_model(
    image_base64: str,
    prompt: str,
    model: str,
    expected_fields: list[str] | None,
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
    expected_fields: list[str] | None = None,
    template_type: str = "business_card",
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
    compressed_base64, mime_type = _prepare_image_for_upload(image_base64)
    file_data_uri = f"data:{mime_type};base64,{compressed_base64}"
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

    # 1) 只有确实包含业务字段时，才把顶层/嵌套 dict 当作结构化结果。
    direct_result = _find_expected_field_result(data, expected_fields)
    if direct_result is not None:
        return direct_result

    # 2) 从 layout_parsing 结果中提取文本。
    text_chunks = _extract_text_chunks(data)
    if template_type == "business_card":
        return _map_business_card_fields(text_chunks)

    if not text_chunks:
        raise RuntimeError("OCR未提取到可结构化文本")

    # layout_parsing 不消费业务 prompt，非名片模板必须二次结构化。
    return _call_glm_structure(text_chunks, prompt, expected_fields)


def _call_glm_structure(
    text_chunks: list[str],
    prompt: str,
    expected_fields: list[str] | None = None,
) -> dict:
    """将 OCR 文本按业务 prompt 结构化为 JSON。"""
    if not GLM_API_KEY:
        raise ValueError("缺少 GLM_API_KEY 环境变量")

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }
    ocr_text = "\n".join(text_chunks)
    payload = {
        "model": STRUCTURE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你只输出一个JSON对象，不要输出解释、Markdown或代码块。",
            },
            {
                "role": "user",
                "content": f"{prompt.strip()}\n\n以下是OCR识别出的文本，请仅基于这些文本提取字段：\n{ocr_text}",
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"GLM结构化请求失败 status={resp.status_code}, body={resp.text[:500]}")

    data = resp.json()
    content = _extract_chat_content(data)
    parsed = _parse_json_from_response(content)
    direct_result = _find_expected_field_result(parsed, expected_fields)
    return direct_result if direct_result is not None else parsed


def _compress_base64_image(
    image_base64: str, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> str:
    compressed, _ = _prepare_image_for_upload(image_base64, max_edge=max_edge, max_bytes=max_bytes)
    return compressed


def _prepare_image_for_upload(
    image_base64: str, max_edge: int = 1280, max_bytes: int = 450 * 1024
) -> tuple[str, str]:
    clean_base64, mime_hint = _strip_data_uri(image_base64)
    try:
        image_bytes = base64.b64decode(clean_base64, validate=True)
    except Exception as e:
        raise InvalidImageError("图片数据不是有效的Base64") from e

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            detected_mime = _image_mime_type(img, mime_hint)
            width, height = img.size
            long_edge = max(width, height)

            # 小且分辨率合理的图片原样上传，保留 PNG/WebP 等真实格式。
            if len(image_bytes) <= 300 * 1024 and long_edge <= max_edge:
                return clean_base64, detected_mime

            img = img.convert("RGB")
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
                    return base64.b64encode(result).decode("utf-8"), "image/jpeg"
                quality -= 10
    except InvalidImageError:
        raise
    except Exception as e:
        raise InvalidImageError("图片数据无法识别为有效图片") from e

    return clean_base64, mime_hint or "image/jpeg"


def _strip_data_uri(image_base64: str) -> tuple[str, str | None]:
    if not isinstance(image_base64, str):
        raise InvalidImageError("图片数据必须是Base64字符串")

    text = image_base64.strip()
    match = re.match(r"^data:([^;,]+);base64,(.*)$", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(2)
        mime_type = match.group(1).lower()
    else:
        mime_type = None

    return "".join(text.split()), mime_type


def _image_mime_type(img: Image.Image, mime_hint: str | None = None) -> str:
    fmt = (img.format or "").upper()
    if fmt in ("JPEG", "JPG"):
        return "image/jpeg"
    if fmt == "PNG":
        return "image/png"
    if fmt == "WEBP":
        return "image/webp"
    if fmt == "GIF":
        return "image/gif"
    return mime_hint if mime_hint and mime_hint.startswith("image/") else "image/jpeg"


def _extract_chat_content(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("GLM结构化响应缺少choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("GLM结构化响应choices格式异常")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("GLM结构化响应缺少message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("GLM结构化响应content不是字符串")
    return content


def _find_expected_field_result(data: dict, expected_fields: list[str] | None) -> dict | None:
    if not expected_fields or not isinstance(data, dict):
        return None

    candidates = [data]
    for key in ("data", "result"):
        value = data.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    for candidate in candidates:
        if any(field in candidate for field in expected_fields):
            return candidate
    return None


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

    def add_text(value: str) -> None:
        for line in value.splitlines():
            line = line.strip()
            if line:
                chunks.append(line)

    def visit(node) -> None:
        if isinstance(node, str):
            add_text(node)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        for key in ("md_results", "content", "text"):
            value = node.get(key)
            if isinstance(value, (str, list, dict)):
                visit(value)

        for key in ("layout_details", "data", "result"):
            value = node.get(key)
            if isinstance(value, (str, list, dict)):
                visit(value)

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
            looks_like_company = _contains_company_token(en_low, company_en_keywords)
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
        looks_like_company = _contains_company_token(en_low, company_en_keywords)
        if not _looks_like_address_phrase(en_name) and not looks_like_company:
            return en_name
    return ""


def _looks_like_address_phrase(value: str) -> bool:
    low = value.lower()
    address_terms = ("factory", "add", "road", "district", "building", "room")
    return any(term in low for term in address_terms)


def _contains_company_token(value: str, keywords: tuple[str, ...]) -> bool:
    tokens = re.findall(r"[a-z]+", value.lower())
    return any(token in keywords for token in tokens)


def _looks_like_valid_cn_name(value: str) -> bool:
    if len(value) < 2 or len(value) > 4:
        return False
    # 常见地名/公司前缀，避免把“上海玖协”当人名
    non_name_prefixes = ("上海", "北京", "广州", "深圳", "中国", "公司", "集团")
    return not any(value.startswith(prefix) for prefix in non_name_prefixes)