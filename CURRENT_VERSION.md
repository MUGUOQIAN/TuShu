# 当前版本记录

- 记录时间: 2026-06-24 11:00:00
- 代码版本: `cursor/critical-bug-investigation-41aa`

## 后端 (ocr-recognize)

- 本地服务地址: `http://127.0.0.1:9000`
- 本地健康检查: `GET /?ping=1`
- 当前版本标识: `ocr-recognize-2026-06-24-v6`
- 主要能力:
  - 接收 `image_base64` + `template_type`
  - 调用 `glm-ocr` 做布局解析
  - 基于解析文本做名片字段映射（姓名/公司/职位/手机/座机/邮箱/地址）
  - 发票/自定义模板在 layout OCR 后调用结构化模型按字段输出 JSON
  - 支持函数计算 `event` 的 `bytes/str/dict` 归一化处理
  - Pillow 新旧版本兼容（`Image.Resampling.LANCZOS` / `Image.LANCZOS` 回退）
  - 保留小 PNG/WebP/JPEG 的真实 data URI MIME，并对高分辨率图片按长边压缩
  - 姓名提取规则优化，避免将公司英文名误判为姓名

## 前端 (tushu)

- 默认 OCR 地址: `https://ocr-recognize-tkvdnxbzyt.cn-shanghai.fcapp.run`
- 可通过 `--dart-define` 覆盖:
  - `OCR_API_BASE_URL=http://127.0.0.1:9000`（本地联调）

## 常用测试命令

- 后端本地健康检查:
  - `python -c "import requests;print(requests.get('http://127.0.0.1:9000?ping=1').text)"`
- 后端本地名片测试（22222.jpg）:
  - `python -c "import base64,requests;from pathlib import Path;img=Path('tushu/test/22222.jpg').read_bytes();b64=base64.b64encode(img).decode('utf-8');print(requests.post('http://127.0.0.1:9000',json={'image_base64':b64,'template_type':'business_card'}).text)"`
- Flutter 本地联调:
  - `cd tushu && flutter test test/api_service_businesscard_test.dart --dart-define=OCR_API_BASE_URL=http://127.0.0.1:9000`
- Flutter 全量测试:
  - `cd tushu && flutter test`

## 说明

- 云端地址是否生效以 `?ping=1` 返回为准。
- 若云端返回 `502` 或非 JSON 页面，前端会出现解析异常，这通常是云端函数/网关配置问题，不是前端请求格式问题。
- 云端当前默认地址：`https://ocr-recognize-tkvdnxbzyt.cn-shanghai.fcapp.run`
- 最近本地验证以本次提交测试记录为准；云端地址是否已部署以 `?ping=1` 返回版本为准。
