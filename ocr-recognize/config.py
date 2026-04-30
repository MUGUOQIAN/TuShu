import os

try:
    # 本地开发时读取 .env；云端若未安装 python-dotenv 也不应启动失败
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# 从云函数环境变量读取，保护密钥安全
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")

# 模型配置（仅GLM）
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "glm-ocr")

# 图片最大Base64长度（约2MB原图）
MAX_IMAGE_SIZE = 2 * 1024 * 1024