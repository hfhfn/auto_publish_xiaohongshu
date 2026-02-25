"""
统一配置模块 — 集中管理所有API密钥和路径参数
"""
import os
from pathlib import Path

# ======================== 项目路径 ========================
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
REF_IMAGE_DIR = OUTPUT_DIR / "ref"        # 搜索到的真实景点参考图
GEN_IMAGE_DIR = OUTPUT_DIR / "generated"  # 图生图生成的图片

# 自动创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)
REF_IMAGE_DIR.mkdir(exist_ok=True)
GEN_IMAGE_DIR.mkdir(exist_ok=True)

# ======================== 智谱AI ========================
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_MODEL = "glm-4.5-flash"
ZHIPU_VLM_MODEL = "glm-4v-flash"          # 视觉模型，用于图片质量评估
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# ======================== 火山引擎 即梦4.0 ========================
def _load_access_key():
    """从 AccessKey.txt 读取火山引擎密钥"""
    ak_file = PROJECT_DIR / "AccessKey.txt"
    ak, sk = "", ""
    with open(ak_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("AccessKeyId:"):
                ak = line.split(":", 1)[1].strip()
            elif line.startswith("SecretAccessKey:"):
                sk = line.split(":", 1)[1].strip()
    return ak, sk

VOLC_AK, VOLC_SK = _load_access_key()
VOLC_API_HOST = "visual.volcengineapi.com"
VOLC_API_URL = f"https://{VOLC_API_HOST}"
JIMENG_REQ_KEY = "jimeng_t2i_v40"

# ======================== Brave Search ========================
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_IMAGE_SEARCH_URL = "https://api.search.brave.com/res/v1/images/search"
BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# ======================== DeepSeek (备用) ========================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ======================== Z-Image (本地) ========================
ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_STEPS = 8
ZIMAGE_STRENGTH = 0.6  # 图生图强度，越低越接近原图

# ======================== 小红书 ========================
XHS_PUBLISH_URL = "https://creator.xiaohongshu.com/publish/publish?source=official"
CHROME_DEBUG_PORT = 9222
CHROME_USER_DATA_DIR = PROJECT_DIR / "chrome_profile"  # 独立Chrome配置目录，避免与日常浏览器冲突
