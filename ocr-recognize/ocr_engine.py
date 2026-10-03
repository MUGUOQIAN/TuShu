import base64
import io
import json
import re
import requests
from PIL import Image
from config import GLM_API_KEY, PRIMARY_MODEL

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow>=9.1
except AttributeError:
    _LANCZOS = Image.LANCZOS  # Pillow<9.1


def call_llm(image_base64: str, prompt: str, model: str = PRIMARY_MODEL) -> dict:
    """
    调用大模型API（仅GLM），返回解析后的JSON结果。
    """
    return _call_model(image_base64, prompt, model)


def _call_model(image_base64: str, prompt: str, model: str) -> dict:
    """实际调用模型API"""
    if model == "glm-ocr":
        return _call_glm(image_base64, prompt)
    else:
        raise ValueError(f"不支持的模型: {model}")


def _call_glm(image_base64: str, prompt: str) -> dict:
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

    # 1) 优先尝试直接JSON结构（兼容不同返回形态）
    if isinstance(data.get("data"), dict):
        return data["data"]
    if isinstance(data.get("result"), dict):
        return data["result"]

    # 2) 从 layout_parsing 结果中提取文本，再做名片字段映射
    text_chunks = _extract_text_chunks(data)
    return _map_business_card_fields(text_chunks)


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


def _extract_text_chunks(data: dict) -> list[str]:
    chunks: list[str] = []

    md_results = data.get("md_results")
    if isinstance(md_results, list):
        for item in md_results:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())

    layout_details = data.get("layout_details")
    if isinstance(layout_details, list):
        for page_items in layout_details:
            if not isinstance(page_items, list):
                continue
            for item in page_items:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(content.strip())

    # 去重保序
    seen = set()
    ordered = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# 日文公司法律形式正好含 2–4 个汉字，姓名正则会整段吞掉，公司关键词又对不上。
_JP_COMPANY_MARKERS = ("株式会社", "有限会社", "合同会社")

# 这些职务正好是 2–4 个汉字，且不含「经理/总监」。长词在前。
_ROLE_TITLES = (
    "名誉主席",
    "执行主席",
    "常务主席",
    "轮值主席",
    "副主席",
    "副社长",
    "副课长",
    "主席",
    "社长",
    "课长",
    "次长",
    "系长",
)
_ROLE_MODIFIERS = ("名誉", "执行", "常务", "轮值", "副")
_ROLE_NAME_BLOCKLIST = (
    "工作室",
    "事务所",
    "办公室",
    "委员会",
    "协会",
    "商会",
    "公司",
    "有限",
    "集团",
    "株式",
    "会社",
)


def _compact_role_text(chunk: str) -> str:
    return re.sub(r"[\s:：/|·•,，;；]+", "", chunk.strip())


def _chunk_is_jp_company(chunk: str) -> bool:
    return any(marker in chunk for marker in _JP_COMPANY_MARKERS)


def _is_only_role_modifiers(text: str) -> bool:
    if not text:
        return True
    rest = text
    while rest:
        matched = ""
        for word in sorted(_ROLE_MODIFIERS, key=len, reverse=True):
            if rest.startswith(word):
                matched = word
                break
        if not matched:
            return False
        rest = rest[len(matched) :]
    return True


def _is_role_person_name(value: str) -> bool:
    if not re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", value):
        return False
    if any(word in value for word in _ROLE_NAME_BLOCKLIST):
        return False
    if any(word in value for word in _ROLE_TITLES):
        return False
    if any(word in value for word in _ROLE_MODIFIERS):
        return False
    return _looks_like_valid_cn_name(value)


def _split_role_line(chunk: str) -> tuple[str, str] | None:
    """识别「社长 / 主席 张三 / 张三课长 / 名誉主席」。

    返回 (职务, 姓名)。整行只有职务时姓名为空。不是这类行时返回 None。
    """
    compact = _compact_role_text(chunk)
    if not compact or _chunk_is_jp_company(compact):
        return None

    for title in sorted(_ROLE_TITLES, key=len, reverse=True):
        if compact.endswith(title) and len(compact) > len(title):
            prefix = compact[: -len(title)]
            if _is_role_person_name(prefix):
                return title, prefix

        index = compact.find(title)
        if index < 0:
            continue
        prefix = compact[:index]
        suffix = compact[index + len(title) :]
        if not _is_only_role_modifiers(prefix):
            continue
        if suffix == "":
            return compact, ""
        if _is_role_person_name(suffix):
            return prefix + title, suffix
    return None


def _fallback_blocks_cn_token(token: str, chunks: list[str]) -> bool:
    """全局回退不能再把公司法律形式或职务的前几个字当成姓名。"""
    if _split_role_line(token) is not None:
        return True
    for chunk in chunks:
        compact = _compact_role_text(chunk)
        if not compact.startswith(token) or compact == token:
            continue
        if _chunk_is_jp_company(chunk):
            return True
        parsed = _split_role_line(chunk)
        if parsed is not None and not parsed[1]:
            return True
    return False


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
    role_title = ""

    company_keywords = (
        "公司",
        "有限",
        "集团",
        "株式会社",
        "合同会社",
        "Co.,Ltd",
        "Co., Ltd",
        "Ltd",
        "Inc",
        "Corporation",
    )
    title_keywords = ("经理", "总监", "主管", "工程师", "销售", "总裁", "主任", "顾问", "Designer", "Manager")
    address_keywords = ("地址", "路", "街", "号", "区", "市", "省", "Address")

    for chunk in chunks:
        parsed_role = _split_role_line(chunk)
        if parsed_role and parsed_role[0] and not role_title:
            role_title = parsed_role[0]
        if not company and any(k in chunk for k in company_keywords):
            company = chunk
        if not title and any(k in chunk for k in title_keywords):
            title = chunk
        if not address and any(k in chunk for k in address_keywords):
            address = chunk

    # 社长/主席不含「经理」。已有销售经理等职位时不要用职务行盖住它。
    if not title:
        title = role_title

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
        parsed_role = _split_role_line(c)
        if parsed_role is not None:
            # 「社长」「主席 张三」不能把职务写成姓名，否则真实姓名被丢掉。
            if parsed_role[1]:
                return parsed_role[1]
            continue
        compact_role = _compact_role_text(c)
        # 「主席办公室」以职务起头，整行不是人名。
        if any(
            compact_role.startswith(title) or compact_role.endswith(title)
            for title in _ROLE_TITLES
        ):
            continue
        # 「株式会社山田制作所」「山田制作所株式会社」整行是公司，不是人名。
        if _chunk_is_jp_company(c):
            continue
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
        if _looks_like_valid_cn_name(cn_name) and not _fallback_blocks_cn_token(cn_name, chunks):
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
    if any(value.startswith(prefix) for prefix in non_name_prefixes):
        return False
    if "会社" in value or "株式" in value:
        return False
    if any(title in value for title in _ROLE_TITLES):
        return False
    return True