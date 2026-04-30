import json
import traceback
from ocr_engine import call_llm
from prompt_templates import TEMPLATE_MAP
from validators import validate_and_clean

SERVICE_VERSION = "ocr-recognize-2026-04-30-v5"


def _normalize_fc_event(event):
    """
    阿里云 HTTP 触发器里 event 可能是 str/bytes 或 body/query 为 bytes，
    统一成 dict，且 body 为 str（JSON 文本）或保持为已解析的 dict。
    """
    if isinstance(event, (bytes, bytearray)):
        event = event.decode("utf-8", errors="replace")
    if isinstance(event, str):
        event = json.loads(event)
    if not isinstance(event, dict):
        raise TypeError(f"event 类型不受支持: {type(event).__name__}")

    qp = event.get("queryParameters")
    if qp is None:
        qp = {}
    elif isinstance(qp, (bytes, bytearray)):
        qp = json.loads(qp.decode("utf-8", errors="replace"))
    elif isinstance(qp, str):
        qp = json.loads(qp) if qp.strip() else {}
    if not isinstance(qp, dict):
        qp = {}

    raw_body = event.get("body", "{}")
    if isinstance(raw_body, (bytes, bytearray)):
        raw_body = raw_body.decode("utf-8", errors="replace")
    if raw_body is None or (isinstance(raw_body, str) and not raw_body.strip()):
        raw_body = "{}"

    return {**event, "queryParameters": qp, "body": raw_body}


def handler(event, context):
    """
    阿里云函数计算HTTP触发器入口。
    接收：{"image_base64": "...", "template_type": "invoice", "custom_fields": "姓名,电话"}
    返回：{"success": true, "data": {"姓名": "张三", ...}, "error": ""}
    """
    try:
        print(f"[handler] start version={SERVICE_VERSION}, event_type={type(event).__name__}")
        event = _normalize_fc_event(event)
        raw_body = event.get("body", "{}")
        body_len = len(raw_body) if isinstance(raw_body, str) else -1
        print(
            f"[handler] normalized event query_keys={list((event.get('queryParameters') or {}).keys())}, body_type={type(raw_body).__name__}, body_len={body_len}"
        )

        # 0. 版本探针：用于确认线上是否命中新部署
        query = event.get("queryParameters", {}) or {}
        if query.get("ping") == "1":
            print("[handler] ping probe hit")
            return _response(200, {"success": True, "version": SERVICE_VERSION})

        # 1. 解析请求
        raw = event.get("body", "{}")
        if isinstance(raw, dict):
            body = raw
        else:
            body = json.loads(raw)
        image_base64 = body.get("image_base64", "")
        template_type = body.get("template_type", "custom")
        custom_fields = body.get("custom_fields", "")
        print(
            f"[handler] request parsed template_type={template_type}, image_base64_len={len(image_base64) if isinstance(image_base64, str) else -1}"
        )

        if not image_base64:
            return _response(400, {"success": False, "error": "缺少图片数据"})

        # 2. 获取模板
        if template_type == "custom":
            if not custom_fields:
                return _response(400, {"success": False, "error": "自定义模板需提供custom_fields"})
            fields = [f.strip() for f in custom_fields.split(",") if f.strip()]
            prompt = TEMPLATE_MAP["custom"]["template"].format(
                fields=", ".join(fields)
            )
        elif template_type in TEMPLATE_MAP:
            config = TEMPLATE_MAP[template_type]
            fields = config["fields"]
            prompt = config["template"]
        else:
            return _response(400, {"success": False, "error": f"未知模板类型: {template_type}"})

        # 3. 调用大模型
        print("[handler] call_llm start")
        raw_result = call_llm(image_base64, prompt)
        print(f"[handler] call_llm done raw_result_type={type(raw_result).__name__}")
        normalized_result = _normalize_llm_result(raw_result)

        # 4. 校验清洗
        cleaned_result = validate_and_clean(normalized_result, fields)
        print("[handler] success")

        return _response(200, {"success": True, "data": cleaned_result})

    except json.JSONDecodeError as e:
        print(f"[handler] json decode error: {e}")
        print(traceback.format_exc())
        return _response(400, {"success": False, "error": f"大模型返回格式异常: {str(e)}"})
    except Exception as e:
        print(f"[handler] unhandled exception: {e}")
        print(traceback.format_exc())
        return _response(500, {"success": False, "error": f"服务内部错误: {str(e)}"})


def _response(status_code: int, body: dict) -> dict:
    """构造阿里云函数HTTP响应"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body, ensure_ascii=False)
    }


def _normalize_llm_result(raw_result) -> dict:
    """
    统一模型输出类型，确保进入校验阶段的是dict。
    支持dict / JSON字符串 / bytes。
    """
    if isinstance(raw_result, dict):
        return raw_result

    if isinstance(raw_result, bytes):
        try:
            raw_result = raw_result.decode("utf-8")
        except UnicodeDecodeError:
            raw_result = raw_result.decode("utf-8", errors="replace")

    if isinstance(raw_result, str):
        text = raw_result.strip()
        # 兼容 ```json ... ``` 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"模型返回无法解析为JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError(f"模型返回JSON不是对象类型: {type(parsed).__name__}")
        return parsed

    raise ValueError(f"模型返回类型不受支持: {type(raw_result).__name__}")