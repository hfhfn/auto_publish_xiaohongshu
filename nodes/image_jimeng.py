"""
即梦4.0 图生图节点 — 使用火山引擎 Visual API 进行图生图
"""
import base64
import hashlib
import hmac
import json
import time
import datetime
import requests
from pathlib import Path
from config import (
    VOLC_AK, VOLC_SK, VOLC_API_HOST, VOLC_API_URL,
    JIMENG_REQ_KEY, GEN_IMAGE_DIR,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sign_request(method: str, action: str, body: bytes, now: datetime.datetime) -> dict:
    """火山引擎 V4 签名"""
    service = "cv"
    region = "cn-north-1"
    version = "2022-08-31"

    date_stamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")

    # 查询参数
    query = f"Action={action}&Version={version}"

    # 规范请求头
    content_type = "application/json"
    host = VOLC_API_HOST
    headers_to_sign = {
        "content-type": content_type,
        "host": host,
        "x-date": amz_date,
    }
    signed_headers = ";".join(sorted(headers_to_sign.keys()))
    canonical_headers = "".join(
        f"{k}:{v}\n" for k, v in sorted(headers_to_sign.items())
    )

    # 规范请求
    payload_hash = _sha256(body)
    canonical_request = "\n".join([
        method, "/", query,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    # 签名字符串
    credential_scope = f"{date_stamp}/{region}/{service}/request"
    string_to_sign = "\n".join([
        "HMAC-SHA256",
        amz_date,
        credential_scope,
        _sha256(canonical_request.encode("utf-8")),
    ])

    # 计算签名
    k_date = _hmac_sha256(VOLC_SK.encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, "request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    auth = (
        f"HMAC-SHA256 Credential={VOLC_AK}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    return {
        "Content-Type": content_type,
        "Host": host,
        "X-Date": amz_date,
        "Authorization": auth,
    }


def _submit_task(prompt: str, image_base64s: list[str] = None) -> str:
    """提交即梦4.0图生图任务，返回task_id"""
    body_dict = {
        "req_key": JIMENG_REQ_KEY,
        "prompt": prompt,
        "model_version": "general_v4.0",
        "return_url": True,
    }
    if image_base64s:
        body_dict["binary_data_base64"] = image_base64s

    body = json.dumps(body_dict).encode("utf-8")
    now = datetime.datetime.utcnow()
    action = "CVSync2AsyncSubmitTask"
    headers = _sign_request("POST", action, body, now)
    url = f"{VOLC_API_URL}/?Action={action}&Version=2022-08-31"

    resp = requests.post(url, headers=headers, data=body, timeout=30)
    if resp.status_code != 200:
        print(f"    ❌ API返回错误 {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 10000 and data.get("code") != 0:
        raise RuntimeError(f"即梦提交任务失败: {data}")

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"未获取到task_id: {data}")

    return task_id


def _get_result(task_id: str, max_wait: int = 120) -> list[str]:
    """轮询获取生图结果，返回图片URL列表"""
    body_dict = {
        "req_key": JIMENG_REQ_KEY,
        "task_id": task_id,
    }
    body = json.dumps(body_dict).encode("utf-8")

    for _ in range(max_wait // 3):
        time.sleep(3)
        now = datetime.datetime.utcnow()
        action = "CVSync2AsyncGetResult"
        headers = _sign_request("POST", action, body, now)
        url = f"{VOLC_API_URL}/?Action={action}&Version=2022-08-31"

        resp = requests.post(url, headers=headers, data=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("data", {}).get("status", "")

        if status == "done":
            data_body = data.get("data", {})
            image_urls = data_body.get("image_urls")
            binary_base64_list = data_body.get("binary_data_base64")
            
            # 兼容既返回URL也可能只返回Base64的情况
            return {
                "urls": image_urls if image_urls else [],
                "base64s": binary_base64_list if binary_base64_list else []
            }
        elif status == "failed":
            raise RuntimeError(f"即梦生图失败: {data}")
        # 其他状态继续等待

    raise TimeoutError(f"即梦生图超时(>{max_wait}s), task_id={task_id}")


def _get_ref_image_base64(image_path: str) -> str:
    """Read the local image, resize if needed, and return as base64 encoded string.

    Volcano Engine API rejects images with dimensions that are too large,
    so we always constrain to 1024x1024 max and re-encode as JPEG.
    """
    from PIL import Image as PILImage
    import io

    img = PILImage.open(image_path)
    # 如果任一边超过1024像素，等比缩小
    if img.width > 1024 or img.height > 1024:
        img.thumbnail((1024, 1024))
    # 转为RGB（去掉alpha通道），统一编码为JPEG
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_CLEANUP_PREFIX = "High quality realistic photograph. Remove all text overlays, watermarks, logos, advertisements and any location name text. "


def generate_image_jimeng(state: dict) -> dict:
    """LangGraph节点：使用即梦4.0进行图生图"""
    image_prompts = state.get("image_prompts", [])
    ref_for_gen = state.get("ref_for_gen", [])
    ref_paths = state.get("ref_image_paths", [])
    topic = state.get("topic", "")

    if not image_prompts:
        image_prompts = [f"A stunning photo of {state['topic']}, realistic photography, 8K"]

    generated_paths = []
    task_dir = GEN_IMAGE_DIR / f"jimeng_{int(time.time())}"
    task_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(image_prompts):
        # 在prompt中强调目标地点，要求去除所有文字和其他地名
        location_hint = f" This photo is specifically about {topic}. " if topic else ""
        full_prompt = _CLEANUP_PREFIX + location_hint + prompt
        print(f"  🎨 即梦4.0 图生图 [{i+1}/{len(image_prompts)}]: {prompt[:50]}...")

        # 使用专门的img2img参考图（与发布原图不同，保持多样性）
        image_base64s = []
        ref_path = None
        if i < len(ref_for_gen) and ref_for_gen[i]:
            ref_path = ref_for_gen[i]
        elif ref_paths:
            ref_path = ref_paths[i % len(ref_paths)]

        if ref_path:
            try:
                b64 = _get_ref_image_base64(ref_path)
                image_base64s = [b64]
                print(f"    📤 参考图已读取转换为 Base64 ({len(b64)//1024}KB)")
            except Exception as e:
                print(f"    ⚠️ 参考图读取失败，使用纯文生图: {e}")

        # 提交任务（无参考图时退化为纯文生图）
        has_ref = bool(image_base64s)
        print(f"    📤 提交任务: prompt长度={len(full_prompt)}, 有参考图={has_ref}")
        task_id = _submit_task(full_prompt, image_base64s if image_base64s else None)
        print(f"    ⏳ 任务已提交: {task_id}")

        # 获取结果
        result_content = _get_result(task_id)
        urls = result_content.get("urls", [])
        base64s = result_content.get("base64s", [])

        # 优先处理URL下载
        for j, img_url in enumerate(urls):
            img_resp = requests.get(img_url, timeout=30)
            img_resp.raise_for_status()
            path = task_dir / f"gen_{i}_url_{j}.jpg"
            path.write_bytes(img_resp.content)
            generated_paths.append(str(path))
            print(f"    ✅ 图片已通过URL下载保存: {path.name}")

        # 处理Base64解析
        for j, b64_str in enumerate(base64s):
            path = task_dir / f"gen_{i}_b64_{j}.jpg"
            path.write_bytes(base64.b64decode(b64_str))
            generated_paths.append(str(path))
            print(f"    ✅ 图片已通过Base64解码保存: {path.name}")

    return {"generated_image_paths": generated_paths}
