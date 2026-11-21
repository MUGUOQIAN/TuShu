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


def _map_business_card_fields(chunks: list[str]) -> dict:
    text = "\n".join(chunks)

    email = _first_match(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    mobile = _extract_mobile(text)
    landline = _extract_landline(text, mobile)

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


# 全角数字和常见横线先折成 ASCII，避免「１３８…」「86－21－…」匹配失败。
_PHONE_FOLD = str.maketrans({
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    "＋": "+",
    "－": "-",
    "—": "-",
    "–": "-",
    "―": "-",
    "−": "-",
    "．": ".",
    "。": ".",
})
# 只允许号码内部的分隔符，最多连续 3 个。不包含换行，避免把相邻两行的无关数字粘成一个电话。
_PHONE_GAP = r"[ \t\u3000.·•()（）\-]{1,3}"
_MOBILE_LOOSE = re.compile(
    rf"(?<!\d)(?:\+?86(?:{_PHONE_GAP})?)?(1[3-9](?:(?:{_PHONE_GAP})?\d){{9}})(?!\d)"
)
_LANDLINE_LOOSE = re.compile(
    rf"(?<!\d)(?P<prefix>\+?86(?:{_PHONE_GAP})?)?(?P<num>0?\d{{2,3}}(?:(?:{_PHONE_GAP})?\d){{7,8}})(?!\d)"
)
_THREE_DIGIT_AREAS = {"010", "020", "021", "022", "023", "024", "025", "027", "028", "029"}


def _fold_phone_text(text: str) -> str:
    return text.translate(_PHONE_FOLD)


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _extract_mobile(text: str) -> str:
    """提取中国大陆手机号。

    连续 11 位仍按原样返回。`138-1234-5678`、`+86 138 1234 5678`、
    `86-13120543237` 这类名片印刷格式会先去掉分隔符再校验。
    """
    folded = _fold_phone_text(text)
    direct = _first_match(r"(?<!\d)1[3-9]\d{9}(?!\d)", folded)
    if direct:
        return direct

    match = _MOBILE_LOOSE.search(folded)
    if not match:
        return ""
    digits = _digits_only(match.group(1))
    if re.fullmatch(r"1[3-9]\d{9}", digits):
        return digits
    return ""


def _format_landline(digits: str) -> str:
    """把区号+本地号规范成 0xx-xxxxxxx。无法可靠拆分时返回空字符串。"""
    if digits.startswith("0"):
        trunk = digits
    else:
        trunk = "0" + digits
    if not re.fullmatch(r"0\d{9,11}", trunk):
        return ""
    if trunk.startswith(("0400", "0800", "400", "800")):
        return ""

    if trunk[:3] in _THREE_DIGIT_AREAS and len(trunk) in (10, 11):
        area, local = trunk[:3], trunk[3:]
    elif len(trunk) == 12:
        area, local = trunk[:4], trunk[4:]
    elif len(trunk) == 11:
        area, local = trunk[:4], trunk[4:]
    else:
        area, local = trunk[:3], trunk[3:]

    if len(local) not in (7, 8):
        return ""
    return f"{area}-{local}"


def _landline_has_signal(raw: str, had_country: bool) -> bool:
    if had_country or raw.startswith("0"):
        return True
    return bool(re.search(r"[-.（）()]", raw))


def _extract_landline(text: str, mobile: str) -> str:
    """提取座机。优先非传真号码。

    接受 `021-12345678`、`86-21-60542168`、`(010) 6234-5678`。
    不把手机号、400/800、纯数字的微博/订单号当成座机。
    """
    folded = _fold_phone_text(text)
    candidates = []
    for match in _LANDLINE_LOOSE.finditer(folded):
        raw = match.group("num")
        had_country = bool(match.group("prefix"))
        if not _landline_has_signal(raw, had_country):
            continue
        digits = _digits_only(raw)
        if digits == mobile or re.fullmatch(r"1[3-9]\d{9}", digits):
            continue
        if digits.startswith(("400", "800")):
            continue
        formatted = _format_landline(digits)
        if not formatted:
            continue
        if _digits_only(formatted).lstrip("0") == mobile:
            continue
        prefix = folded[max(0, match.start() - 16) : match.start()]
        is_fax = bool(re.search(r"fax|传真", prefix, re.IGNORECASE))
        candidates.append((is_fax, match.start(), formatted))

    if not candidates:
        return ""
    plain = [item for item in candidates if not item[0]]
    chosen = plain or candidates
    chosen.sort(key=lambda item: item[1])
    return chosen[0][2]


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