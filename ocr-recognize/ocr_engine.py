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
    chunks = [_normalize_ocr_text(chunk) for chunk in chunks]
    text = "\n".join(chunks)

    email = _extract_email(text)

    # 名片常把手机写成 138-0013-8000 / +86 138 0013 8000，不能要求 11 位连写。
    mobile = _extract_mobile(text)
    landline = _extract_landline(text)
    if re.sub(r"\D", "", landline) == re.sub(r"\D", "", mobile):
        landline = ""
    # 名片常把 400/800 热线当主电话；不能因为不是 0 开头区号就整段丢掉。
    if not landline:
        landline = _extract_hotline(text)

    company = ""
    name = ""
    title = ""
    address = ""

    company_chunks: list[str] = []
    title_chunks: list[str] = []
    # (is_company, is_strong, chunk)
    address_chunks: list[tuple[bool, bool, str]] = []

    for chunk in chunks:
        is_company_chunk = _is_company_line(chunk)
        if is_company_chunk:
            company_chunks.append(chunk)
        # 公司名常含“销售/顾问”，不能抢占独立职位行。
        is_title_chunk = _is_title_line(chunk) and not is_company_chunk
        if is_title_chunk:
            title_chunks.append(chunk)
        # 二字中文名（如“张路”）含路/街/号，不能抢占真实地址行。
        # 公司名常含“市/路/区/省”，优先使用非公司地址行。
        # 职位行常含“市/区”（区域经理/市场总监），手机号/工号含“号”，
        # 都不能抢占真实街道地址。
        # 英文 Road/Street 必须整词匹配，避免 Broadway/Groom 子串误判。
        # 「道」已是强地址信号，但漏进行检测时，滨江大道1888 会整行丢失。
        if _looks_like_address_line(chunk) and not _is_two_char_cn_name(chunk):
            strong = _has_strong_address_signal(chunk)
            if not strong and (is_title_chunk or _is_contact_label_line(chunk)):
                continue
            address_chunks.append((is_company_chunk, strong, chunk))

    company = company_chunks[0] if company_chunks else ""
    title = title_chunks[0] if title_chunks else ""
    dedicated_strong = [c for is_company_chunk, strong, c in address_chunks if not is_company_chunk and strong]
    dedicated_addresses = [c for is_company_chunk, _strong, c in address_chunks if not is_company_chunk]
    if dedicated_strong:
        address = dedicated_strong[0]
    elif dedicated_addresses:
        address = dedicated_addresses[0]
    elif address_chunks:
        # 仅有“公司名+地址”混排行时，保留该行以免地址全空。
        address = address_chunks[0][2]
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


_FULLWIDTH_DIGIT_TABLE = str.maketrans("０１２３４５６７８９", "0123456789")


def _normalize_ocr_text(value: str) -> str:
    # 中文 OCR 常输出全角数字与弯引号，不归一会丢掉手机号或截断 O'Brien。
    return (
        value.translate(_FULLWIDTH_DIGIT_TABLE)
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("＇", "'")
    )


# 名片座机常写成 (021) 1234-5678 / 021.12345678 / 138/0013/8000，分隔符还要覆盖括号、点号和斜杠。
_PHONE_SEP = r"[\s\-－—–.()（）/]"
_LANDLINE_PATTERN = re.compile(
    rf"(?<!\d)(0\d{{2,3}}(?:{_PHONE_SEP}*\d){{7,8}})(?!\d)"
)
_FAX_LABELS = ("传真", "fax")
_PHONE_LABELS = ("电话", "座机", "tel", "phone")


def _extract_mobile(text: str) -> str:
    match = re.search(
        rf"(?<!\d)(?:\+?86{_PHONE_SEP}*)?1[3-9](?:{_PHONE_SEP}?\d){{9}}(?!\d)",
        text,
    )
    if not match:
        return ""
    digits = re.sub(r"\D", "", match.group(0))
    if digits.startswith("86") and len(digits) >= 13:
        digits = digits[2:]
    if len(digits) == 11 and digits[0] == "1" and digits[1] in "3456789":
        return digits
    return ""


def _normalize_landline(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    # 00 是国际字冠，0086-138-0013-8000 不能当成座机。
    if (
        not digits.startswith("0")
        or digits.startswith("00")
        or len(digits) < 10
        or len(digits) > 12
    ):
        return ""
    return re.sub(rf"{_PHONE_SEP}+", "-", raw).strip("-")


def _extract_hotline(text: str) -> str:
    match = re.search(
        rf"(?<!\d)[48]00(?:{_PHONE_SEP}?\d){{7}}(?!\d)",
        text,
    )
    if not match:
        return ""
    digits = re.sub(r"\D", "", match.group(0))
    if len(digits) == 10 and digits[:3] in ("400", "800"):
        return re.sub(rf"{_PHONE_SEP}+", "-", match.group(0)).strip("-")
    return ""


def _extract_email(text: str) -> str:
    # 中文 OCR 常把 @ 识别成全角 ＠，或在 @ 两侧插入空格。
    normalized = text.replace("＠", "@").replace("．", ".")
    normalized = re.sub(r"\s*@\s*", "@", normalized)
    return _first_match(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", normalized)


def _line_around(text: str, index: int) -> str:
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _is_fax_only_line(line: str) -> bool:
    low = line.lower()
    has_fax = any(label in low for label in _FAX_LABELS)
    has_phone = any(label in low for label in _PHONE_LABELS)
    return has_fax and not has_phone


def _extract_landline(text: str) -> str:
    matches = list(_LANDLINE_PATTERN.finditer(text))
    if not matches:
        return ""
    preferred = [m for m in matches if not _is_fax_only_line(_line_around(text, m.start()))]
    chosen = (preferred or matches)[0]
    return _normalize_landline(chosen.group(0))


def _extract_name(chunks: list[str]) -> str:
    text = "\n".join(chunks)

    # 先在单行里找疑似姓名，避免全局命中公司简称（如“上海玖协”）
    skip_keywords = (
        "公司",
        "有限",
        "集团",
        "事务所",
        "工作室",
        "研究所",
        "大学",
        "地址",
        "电话",
        "手机",
        "邮箱",
        "@",
        "＠",
        "www",
        ".com",
        "factory",
    )
    # 路/街/号/道/大厦/楼/巷/弄/座/栋/层/室/园/广场/中心只跳过非二字中文名，避免“张路”“李园”被当成地址行丢掉。
    cn_address_markers = (
        "路", "街", "号", "道", "大厦", "楼", "巷", "弄", "座", "栋", "层", "室", "园", "广场", "中心"
    )
    # 英文地址/联系词必须按整词匹配，否则 Addison/Broadway/Ismail
    # 会被 add/road/mail 子串误跳过。
    skip_address_words = (
        "add",
        "address",
        "road",
        "street",
        "avenue",
        "ave",
        "district",
        "room",
        "building",
        "floor",
        "mail",
        "email",
        "plaza",
        "square",
        "parkway",
        "circle",
        "center",
        "tower",
    )
    company_en_keywords = _EN_COMPANY_WORDS + _EN_WEAK_COMPANY_WORDS + (
        "machinery",
        "shanghai",
        "jiuxie",
    )
    cn_name_pattern = re.compile(r"[\u4e00-\u9fa5]{2,4}")
    # 不能用 \s：全局回退文本按行拼接后，John A. Smith 会把 Director/Acme 拼进姓名。
    en_name_pattern = _EN_NAME_PATTERN
    cn_en_combo_pattern = _CN_EN_COMBO_PATTERN

    english_name_candidate = ""

    for i, chunk in enumerate(chunks):
        c = chunk.strip()
        if not c:
            continue
        low = c.lower()
        if _is_company_line(c):
            continue
        if any(k.lower() in low for k in skip_keywords):
            continue
        if any(m in c for m in cn_address_markers) and not _is_two_char_cn_name(c):
            continue
        if _has_whole_word(low, skip_address_words):
            continue
        # 88 West Rd / 15 Oak Dr 带门牌缩写，不能当姓名。
        if _has_numbered_street_type(c):
            continue
        if _is_title_line(c):
            # 名片常把职位和姓名印在同一行（销售总监 张三 / Director Jane Doe）。
            mixed_name = _extract_name_from_title_line(c)
            if mixed_name:
                return mixed_name
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
                near_title = _is_title_line(prev_chunk) or _is_title_line(next_chunk)
                if near_title:
                    return en_name
                if not english_name_candidate:
                    english_name_candidate = en_name

    # 回退：全局搜索
    if english_name_candidate:
        return english_name_candidate

    for cn in cn_name_pattern.finditer(text):
        cn_name = cn.group(0)
        if _looks_like_valid_cn_name(cn_name) and not any(
            keyword in cn_name for keyword in _CN_TITLE_KEYWORDS
        ):
            return cn_name
    for en in en_name_pattern.finditer(text):
        en_name = en.group(0)
        en_low = en_name.lower()
        looks_like_company = _has_whole_word(en_low, company_en_keywords)
        if (
            not _looks_like_address_phrase(en_name)
            and not looks_like_company
            and not _is_title_line(en_name)
        ):
            return en_name
    return ""


# 名片英文名常全大写（JOHN SMITH），也常带中间名缩写（John A. Smith）、
# 连字符（Mary-Jane）或撇号（Patrick O'Brien）。
# 只用非换行空白，避免 "John A. Smith\nDirector\nAcme" 被拼成跨行垃圾姓名。
_EN_NAME_WS = r"[^\S\n]"
_EN_GIVEN_NAME = r"[A-Z](?:[A-Za-z]+|'[A-Za-z]+)(?:-[A-Z](?:[A-Za-z]+|'[A-Za-z]+))?"
_EN_NAME_PART = (
    rf"(?:{_EN_NAME_WS}+{_EN_GIVEN_NAME}|{_EN_NAME_WS}+[A-Z](?!['A-Za-z])\.?)"
)
_EN_NAME_PATTERN = re.compile(rf"\b{_EN_GIVEN_NAME}(?:{_EN_NAME_PART}){{1,2}}\b")
_CN_EN_COMBO_PATTERN = re.compile(
    rf"([\u4e00-\u9fa5]{{2,4}}){_EN_NAME_WS}+({_EN_GIVEN_NAME}(?:{_EN_NAME_PART}){{1,2}})"
)


def _looks_like_address_phrase(value: str) -> bool:
    address_terms = (
        "factory",
        "add",
        "address",
        "road",
        "street",
        "avenue",
        "ave",
        "district",
        "building",
        "room",
        "floor",
        "plaza",
        "square",
        "parkway",
        "circle",
        "center",
        "tower",
    )
    return _has_whole_word(value.lower(), address_terms)


_CN_ADDRESS_KEYWORDS = (
    "地址",
    "路",
    "街",
    "号",
    "区",
    "市",
    "省",
    "道",
    "大厦",
    "楼",
    "巷",
    "弄",
    "座",
    "栋",
    "层",
    "室",
    "园",
    "广场",
    "中心",
)
_EN_ADDRESS_WORDS = (
    "add",
    "address",
    "road",
    "street",
    "avenue",
    "ave",
    "district",
    "building",
    "room",
    "floor",
    "center",
    "tower",
)
# Lane/Drive/Rd/St 等必须带门牌数字，避免把姓名 Lane Cooper / St John 当地址。
_EN_STREET_TYPES = (
    "lane",
    "drive",
    "way",
    "suite",
    "unit",
    "blvd",
    "boulevard",
    "court",
    "place",
    "rd",
    "st",
    "dr",
    "ln",
    "ct",
    "pl",
    "hwy",
    "highway",
    "pkwy",
    "parkway",
    "circle",
    "crescent",
    "terrace",
    "plaza",
    "square",
    "box",
    "rm",
    "apt",
    "apartment",
)
_CN_COMPANY_KEYWORDS = (
    "公司",
    "有限",
    "集团",
    "事务所",
    "工作室",
    "研究所",
    "大学",
    "医院",
    "银行",
    "学校",
    "学院",
    "中学",
    "小学",
    "协会",
    "基金会",
    "分行",
    "股份",
    "控股",
    "实业",
    "研究院",
    "设计院",
    "诊所",
    "幼儿园",
    "出版社",
    "商会",
)
# 科技/软件/网络常出现在公司名，也会出现在“科技经理/软件工程师”这类职位行，不能当强公司后缀。
# 厂/保险/证券/基金/制药 同理：平安保险、中信证券是公司，保险顾问、证券分析师、厂长只是职位。
# 地产/资本/物流/快递/咨询/建设/酒店/航空 同理：万科地产、红杉资本、京东物流是公司，
# 投资经理、物流经理、咨询顾问、建设路（含路）只是职位或地址。
# 人寿/信托/医疗/超市/食品 同理：新华人寿、中信信托、迈瑞医疗、永辉超市是公司，
# 金融经理、医疗顾问、生物工程师、食品路（含路）只是职位或地址。
_CN_WEAK_COMPANY_KEYWORDS = (
    "科技",
    "软件",
    "网络",
    "厂",
    "保险",
    "证券",
    "基金",
    "制药",
    "药业",
    "传媒",
    "贸易",
    "地产",
    "房地产",
    "资本",
    "投资",
    "物流",
    "快递",
    "速运",
    "咨询",
    "建设",
    "酒店",
    "航空",
    "汽车",
    "广告",
    "化工",
    "石油",
    "石化",
    "电网",
    "电力",
    "置地",
    "置业",
    "海运",
    "航运",
    "物业",
    "教育",
    "电信",
    "通信",
    "电子",
    "人寿",
    "信托",
    "期货",
    "租赁",
    "担保",
    "金融",
    "医疗",
    "医药",
    "生物",
    "超市",
    "百货",
    "餐饮",
    "食品",
    "酒业",
    "矿业",
    "钢铁",
    "半导体",
    "新能源",
    "纸业",
)
_EN_COMPANY_PHRASES = ("co.,ltd", "co., ltd")
# Co/Company/Factory 会出现在 Acme Co 这类公司行，也会出现在 Co-Founder / Company Director / Factory Manager。
# Capital/Logistics/Energy 会出现在 Sequoia Capital / DHL Logistics，也会出现在 Investment Director。
# Ventures/Pharma/Partners/Systems 会出现在 Sequoia Ventures / Acme Pharma / Smith Partners / Cisco Systems，
# 也会出现在 Pharma Director / Systems Engineer / Managing Partner。
_EN_WEAK_COMPANY_WORDS = (
    "co",
    "company",
    "factory",
    "securities",
    "insurance",
    "fund",
    "funds",
    "media",
    "capital",
    "logistics",
    "energy",
    "motors",
    "properties",
    "property",
    "investment",
    "investments",
    "express",
    "hotels",
    "hotel",
    "airlines",
    "airline",
    "consulting",
    "telecom",
    "shipping",
    "chemical",
    "chemicals",
    "electronics",
    "communications",
    "realty",
    "construction",
    "advertising",
    "education",
    "estate",
    "ventures",
    "pharma",
    "biotech",
    "partners",
    "mining",
    "foods",
    "trading",
    "healthcare",
    "medical",
    "systems",
)
_EN_COMPANY_WORDS = (
    "ltd",
    "inc",
    "corporation",
    "llc",
    "llp",
    "limited",
    "gmbh",
    "corp",
    "group",
    "holdings",
    "hospital",
    "university",
    "school",
    "institute",
    "bank",
    "plc",
    "foundation",
    "association",
    "college",
    "academy",
    "technologies",
    "studio",
    "studios",
    "labs",
    "solutions",
    "enterprises",
)
_CN_TITLE_KEYWORDS = (
    "经理",
    "总监",
    "主管",
    "工程师",
    "销售",
    "总裁",
    "主任",
    "顾问",
    "董事长",
    "创始人",
    "设计师",
    "架构师",
    "合伙人",
    "律师",
    "教授",
    "医师",
    "医生",
    "校长",
    "老师",
    "行长",
    "院长",
    "秘书长",
    "园长",
    "研究员",
    "博士",
    "理事长",
    "厂长",
    "分析师",
    "助理",
    "专员",
    "代表",
    "会计",
    "出纳",
    "运营",
    "编辑",
    "记者",
    "书记",
    "处长",
    "科长",
    "局长",
    "部长",
    "秘书",
    "文员",
    "店长",
    "技师",
    "监理",
    "护士",
    "药师",
)
_EN_TITLE_KEYWORDS = (
    "designer",
    "manager",
    "director",
    "ceo",
    "cto",
    "cfo",
    "coo",
    "cmo",
    "cio",
    "vp",
    "president",
    "founder",
    "partner",
    "engineer",
    "consultant",
    "professor",
    "architect",
    "analyst",
    "chairman",
    "officer",
    "scientist",
    "lawyer",
    "attorney",
    "representative",
    "specialist",
    "coordinator",
    "executive",
    "associate",
    "accountant",
    "teacher",
    "assistant",
    "secretary",
    "supervisor",
    "counsel",
    "advisor",
    "adviser",
    "principal",
    "developer",
    "nurse",
)
_TITLE_LEFTOVER_WORDS = (
    "chief",
    "technology",
    "software",
    "senior",
    "junior",
    "vice",
    "assistant",
    "associate",
    "head",
    "lead",
    "of",
    "and",
    "the",
    "deputy",
    "co",
    "company",
    "factory",
    "securities",
    "insurance",
    "fund",
    "funds",
    "media",
    "sales",
    "marketing",
    "customer",
    "business",
    "operations",
    "finance",
    "product",
    "project",
    "account",
    "human",
    "resources",
    "hr",
    "administrative",
    "capital",
    "logistics",
    "energy",
    "investment",
    "consulting",
    "education",
    "ventures",
    "pharma",
    "biotech",
    "partners",
    "mining",
    "foods",
    "trading",
    "healthcare",
    "medical",
    "systems",
    "legal",
    "office",
    "general",
    "行政",
    "客户",
    "业务",
    "市场",
    "人力",
    "财务",
    "技术",
    "运营",
    "高级",
    "资深",
    "初级",
    "副",
    "党委",
    "人事",
    "前台",
    "维修",
    "工程",
    "办公室",
    "金融",
    "医疗",
    "生物",
)


def _is_company_line(chunk: str) -> bool:
    has_strong_cn = any(k in chunk for k in _CN_COMPANY_KEYWORDS)
    low = chunk.lower()
    has_en = any(phrase in low for phrase in _EN_COMPANY_PHRASES) or _has_whole_word(
        low, _EN_COMPANY_WORDS
    )
    if has_strong_cn or has_en:
        return True
    # “杭州云启科技 / 平安保险 / 万科地产 / Acme Co / Sequoia Capital”是公司；
    # “科技经理 / 保险顾问 / 投资经理 / Co-Founder / Investment Director”
    # 只是职位，不能抢占公司字段。
    # “科技大厦 / 张江科技园 / 建设路88号”是地址，不能因为含弱公司词就被当成公司。
    if (
        (
            any(k in chunk for k in _CN_WEAK_COMPANY_KEYWORDS)
            or _has_whole_word(low, _EN_WEAK_COMPANY_WORDS)
        )
        and not _is_title_line(chunk)
        and not _has_strong_address_signal(chunk)
    ):
        return True
    return False


def _is_title_line(chunk: str) -> bool:
    if any(k in chunk for k in _CN_TITLE_KEYWORDS):
        return True
    return _has_whole_word(chunk.lower(), _EN_TITLE_KEYWORDS)


def _extract_name_from_title_line(chunk: str) -> str:
    """从“职位+姓名”混排行里抽出姓名，避免整行被当成职位后姓名全空。"""
    stripped = chunk
    for keyword in sorted(_CN_TITLE_KEYWORDS, key=len, reverse=True):
        stripped = stripped.replace(keyword, " ")
    for keyword in _EN_TITLE_KEYWORDS:
        stripped = re.sub(rf"\b{re.escape(keyword)}\b", " ", stripped, flags=re.I)
    stripped = re.sub(r"[\s,;:：、/|·•\-]+", " ", stripped).strip()
    if not stripped:
        return ""

    leftover = [
        token
        for token in stripped.split()
        if token.lower() not in _TITLE_LEFTOVER_WORDS
    ]
    remainder = " ".join(leftover).strip()
    if not remainder:
        return ""

    if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", remainder) and _looks_like_valid_cn_name(
        remainder
    ):
        return remainder
    english = _EN_NAME_PATTERN.fullmatch(remainder)
    if english and not _looks_like_address_phrase(english.group(0)):
        return english.group(0)

    cn_tokens = [
        token
        for token in leftover
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", token)
        and _looks_like_valid_cn_name(token)
    ]
    if cn_tokens:
        return cn_tokens[-1]

    english_search = _EN_NAME_PATTERN.search(remainder)
    if (
        english_search
        and not _looks_like_address_phrase(english_search.group(0))
        and not _is_title_line(english_search.group(0))
    ):
        return english_search.group(0)
    return ""


def _has_numbered_street_type(chunk: str) -> bool:
    return bool(re.search(r"\d", chunk)) and _has_whole_word(chunk.lower(), _EN_STREET_TYPES)


def _looks_like_address_line(chunk: str) -> bool:
    if any(k in chunk for k in _CN_ADDRESS_KEYWORDS) or "Address" in chunk:
        return True
    if _has_whole_word(chunk.lower(), _EN_ADDRESS_WORDS):
        return True
    return _has_numbered_street_type(chunk)


def _has_whole_word(value: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(keyword.lower())}\b", value) for keyword in keywords)


def _looks_like_valid_cn_name(value: str) -> bool:
    if len(value) < 2 or len(value) > 4:
        return False
    # 常见地名/公司前缀，避免把“上海玖协”“杭州云启”当人名
    non_name_prefixes = (
        "上海",
        "北京",
        "广州",
        "深圳",
        "中国",
        "公司",
        "集团",
        "杭州",
        "成都",
        "武汉",
        "南京",
        "苏州",
        "天津",
        "重庆",
        "西安",
        "长沙",
        "青岛",
        "厦门",
        "宁波",
        "无锡",
        "合肥",
        "郑州",
        "济南",
        "福州",
        "大连",
        "沈阳",
        "哈尔滨",
        "昆明",
        "南宁",
        "南昌",
        "石家庄",
        "太原",
        "长春",
        "贵阳",
        "海口",
        "兰州",
        "银川",
        "西宁",
        "拉萨",
        "香港",
        "澳门",
        "台湾",
        "江苏",
        "浙江",
        "广东",
        "四川",
        "湖北",
        "湖南",
        "河南",
        "河北",
        "山东",
        "山西",
        "安徽",
        "福建",
        "江西",
        "辽宁",
        "吉林",
        "黑龙江",
        "云南",
        "贵州",
        "海南",
        "陕西",
        "甘肃",
        "青海",
        "内蒙古",
        "新疆",
        "西藏",
        "广西",
        "宁夏",
    )
    if any(value.startswith(prefix) for prefix in non_name_prefixes):
        return False
    if value in _CN_COMPANY_KEYWORDS or value in _CN_WEAK_COMPANY_KEYWORDS:
        return False
    # 3 字及以上含路/街/号/道/大厦/楼/座/栋/巷/弄/层/室/园/广场更像地址，二字名（张路/李园）仍保留
    if len(value) >= 3 and any(
        marker in value
        for marker in (
            "路",
            "街",
            "号",
            "道",
            "大厦",
            "楼",
            "座",
            "栋",
            "巷",
            "弄",
            "层",
            "室",
            "园",
            "广场",
            "中心",
            "医院",
            "银行",
            "学校",
            "学院",
            "中学",
            "小学",
            "协会",
            "基金",
            "科技",
            "股份",
            "控股",
            "诊所",
            "幼儿",
            "研究",
            "保险",
            "证券",
            "制药",
            "药业",
            "传媒",
            "贸易",
            "厂",
            "地产",
            "资本",
            "投资",
            "物流",
            "快递",
            "速运",
            "咨询",
            "建设",
            "酒店",
            "航空",
            "汽车",
            "广告",
            "化工",
            "石油",
            "石化",
            "电网",
            "电力",
            "置地",
            "置业",
            "海运",
            "航运",
            "物业",
            "教育",
            "电信",
            "通信",
            "电子",
            "人寿",
            "信托",
            "期货",
            "租赁",
            "担保",
            "金融",
            "医疗",
            "医药",
            "生物",
            "超市",
            "百货",
            "餐饮",
            "食品",
            "酒业",
            "矿业",
            "钢铁",
            "半导体",
            "新能源",
            "纸业",
        )
    ):
        return False
    return True


def _is_two_char_cn_name(value: str) -> bool:
    text = value.strip()
    return bool(re.fullmatch(r"[\u4e00-\u9fa5]{2}", text)) and _looks_like_valid_cn_name(text)


_CONTACT_LABELS = ("手机", "电话", "传真", "微信", "工号", "QQ")
_EN_CONTACT_LABELS = ("tel", "fax", "mobile", "phone")
_STRONG_ADDRESS_MARKERS = ("地址", "Address", "路", "街", "道", "大厦", "座", "栋", "巷", "弄", "园", "广场", "中心")
_STRONG_EN_ADDRESS_WORDS = ("address", "road", "street", "avenue", "ave", "building", "floor", "center", "tower")


def _is_contact_label_line(chunk: str) -> bool:
    if any(label in chunk for label in _CONTACT_LABELS):
        return True
    return _has_whole_word(chunk.lower(), _EN_CONTACT_LABELS)


def _has_strong_address_signal(chunk: str) -> bool:
    if any(marker in chunk for marker in _STRONG_ADDRESS_MARKERS):
        return True
    if _has_whole_word(chunk.lower(), _STRONG_EN_ADDRESS_WORDS):
        return True
    if _has_numbered_street_type(chunk):
        return True
    if re.search(r"\d\s*[层室楼]|[层室楼]\s*\d", chunk) and not _is_contact_label_line(chunk):
        return True
    # “世纪大道1号”之外，纯门牌“88号”也算地址；手机号/工号等联系行除外。
    if "号" in chunk and re.search(r"\d", chunk) and not _is_contact_label_line(chunk):
        return True
    return False