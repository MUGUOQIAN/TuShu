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


# 职务行里常见的职称。长词必须排在短词前面，避免「销售经理」被拆成「销售」+「经理」。
_TITLE_LINE_TITLES = (
    "副总经理",
    "总经理",
    "销售经理",
    "销售总监",
    "销售主管",
    "总工程师",
    "经理",
    "总监",
    "主管",
    "工程师",
    "销售",
    "总裁",
    "主任",
    "顾问",
)
# 只从姓名两侧剥掉这些修饰，避免「高级销售经理张三」把修饰留在姓名里。
_TITLE_LINE_MODIFIERS = (
    "大客户",
    "办公室",
    "事业部",
    "华东区",
    "华南区",
    "华北区",
    "华中区",
    "西南区",
    "西北区",
    "东北区",
    "高级",
    "资深",
    "见习",
    "初级",
    "中级",
    "助理",
    "常务",
    "执行",
    "副总",
    "客户",
    "区域",
    "市场",
    "产品",
    "项目",
    "渠道",
    "技术",
    "财务",
    "人事",
    "行政",
    "运营",
    "商务",
    "法律",
    "战略",
    "管理",
    "投资",
    "国内",
    "海外",
    "华南",
    "华东",
    "华北",
    "华中",
    "西南",
    "西北",
    "东北",
    "全国",
    "客服",
    "部门",
    "本部",
    "热线",
    "专线",
)
_TITLE_LINE_CONNECTORS = ("部", "处", "科", "组", "中心")
# 公司简称紧挨职位时不能当人名，例如「久协机械销售经理」。
_TITLE_LINE_NOT_NAME = (
    "机械",
    "科技",
    "设备",
    "贸易",
    "实业",
    "电子",
    "网络",
    "信息",
    "工程",
    "制造",
    "国际",
    "股份",
    "产业",
    "咨询",
    "集团",
    "公司",
    "有限",
    "热线",
    "专线",
    "微信",
    "邮箱",
    "网址",
    "部门",
    "本部",
    "分机",
    "总机",
    "直线",
    "客服",
)
_EN_TITLE_PHRASES = (
    "general manager",
    "sales manager",
    "sales director",
    "general director",
    "manager",
    "director",
    "designer",
    "senior",
    "vice",
    "deputy",
    "executive",
    "assistant",
    "chief",
)


def _strip_title_edges(text: str) -> str:
    words = sorted(set(_TITLE_LINE_TITLES + _TITLE_LINE_MODIFIERS + _TITLE_LINE_CONNECTORS), key=len, reverse=True)
    changed = True
    while text and changed:
        changed = False
        for word in words:
            if text.startswith(word):
                text = text[len(word):]
                changed = True
                break
            if text.endswith(word):
                text = text[: -len(word)]
                changed = True
                break
    # 「李总」要保留；「张伟总」「副」这种贴在职位上的称呼才剥掉。
    changed = True
    while text and changed:
        changed = False
        if text in ("副", "总", "副总"):
            return ""
        for char in ("副", "总"):
            if text.endswith(char) and len(text) - 1 >= 2:
                text = text[:-1]
                changed = True
                break
            if text.startswith(char) and len(text) - 1 >= 2:
                text = text[1:]
                changed = True
                break
    return text


def _is_title_line_person_name(value: str) -> bool:
    if not re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", value):
        return False
    if not _looks_like_valid_cn_name(value):
        return False
    return not any(fragment in value for fragment in _TITLE_LINE_NOT_NAME)


def _cn_name_from_title_line(chunk: str) -> str:
    compact = re.sub(r"[^\u4e00-\u9fa5]", "", chunk)
    if not compact:
        return ""
    for title in sorted(set(_TITLE_LINE_TITLES), key=len, reverse=True):
        index = compact.find(title)
        if index < 0:
            continue
        prefix = _strip_title_edges(compact[:index])
        suffix = _strip_title_edges(compact[index + len(title):])
        if prefix and suffix:
            continue
        candidate = prefix or suffix
        # 整段就是职务时停住，避免从「总经理」里再拆出「总」。
        if not candidate:
            return ""
        if _is_title_line_person_name(candidate):
            return candidate
    return ""


def _english_name_blocked(value: str) -> bool:
    blocked = (
        "factory",
        "add",
        "road",
        "district",
        "building",
        "room",
        "co",
        "ltd",
        "inc",
        "corporation",
        "machinery",
        "shanghai",
        "jiuxie",
        "company",
    )
    low = value.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", low) for word in blocked)


def _english_name_from_title_line(chunk: str) -> str:
    text = re.sub(r"[\u4e00-\u9fa5]+", " ", chunk)
    for phrase in sorted(_EN_TITLE_PHRASES, key=len, reverse=True):
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:Mr|Ms|Mrs|Miss|Dr)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}", text):
        return ""
    if _english_name_blocked(text):
        return ""
    return text


def _name_from_title_line(chunk: str) -> str:
    """从「张三 总经理」这类职务行取出人名。整行只有职务时返回空字符串。"""
    if any(keyword in chunk for keyword in ("公司", "有限", "集团", "株式", "会社", "Ltd", "Inc", "Corporation")):
        return ""
    cn_name = _cn_name_from_title_line(chunk)
    en_name = _english_name_from_title_line(chunk)
    if cn_name and en_name:
        return f"{cn_name} ({en_name})"
    return cn_name or en_name


def _extract_name(chunks: list[str]) -> str:
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
    title_line_name = ""
    title_lines = set()

    for i, chunk in enumerate(chunks):
        c = chunk.strip()
        if not c:
            continue
        low = c.lower()
        if any(k.lower() in low for k in skip_keywords):
            continue
        if any(k in c for k in title_keywords):
            # 公司行已经在上面跳过。这里保留「张三 总经理」里的人名，
            # 否则全局回退会把公司名或职务的前几个字当成姓名。
            title_lines.add(c)
            if not title_line_name:
                title_line_name = _name_from_title_line(c)
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

    # 回退：全局搜索。职务行里已经抽出的人名优先于公司名片段。
    # 职务行本身不再参与回退，避免「高级销售经理」被截成「高级销售」。
    if english_name_candidate:
        return english_name_candidate
    if title_line_name:
        return title_line_name

    fallback_text = "\n".join(chunk for chunk in chunks if chunk.strip() not in title_lines)
    cn = cn_name_pattern.search(fallback_text)
    if cn:
        cn_name = cn.group(0)
        if _looks_like_valid_cn_name(cn_name):
            return cn_name
    en = en_name_pattern.search(fallback_text)
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