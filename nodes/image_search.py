"""
图片搜索节点 — 使用Brave Search API按prompt分组搜索真实景点图片并下载，
通过 GLM VLM 进行图片质量评估。6组搜索并发执行，VLM调用限速保护。
"""
import base64
import io
import re
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from config import (
    BRAVE_API_KEY, BRAVE_IMAGE_SEARCH_URL, REF_IMAGE_DIR,
    ZHIPU_API_KEY, ZHIPU_VLM_MODEL, ZHIPU_API_URL,
)

# 最小图片尺寸和文件大小（基础预过滤，避免对明显无用的图片调VLM浪费token）
_MIN_DIMENSION = 400
_MIN_FILE_SIZE = 50 * 1024  # 50KB

# 每条query下载的候选数（VLM评估后只保留最好的几张）
_CANDIDATES_PER_QUERY = 6
# 每条query最终保留的最佳图片数（前2张发布用，第3张做img2img参考）
_KEEP_PER_QUERY = 3

# VLM并发控制：最多2个同时调用，避免API限速
_VLM_SEMAPHORE = threading.Semaphore(2)


def _vlm_evaluate_image(image_path: str, query: str) -> float:
    """调用 GLM VLM 评估图片质量，返回1.0-10.0分（含0.5小数）。
    内置429限速重试。
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")

    prompt = (
        f'请严格评估这张图片作为旅游博主发布素材的质量。搜索关键词："{query}"\n\n'
        "请从以下6个维度逐一打分，每个维度1-10分，允许0.5的小数（如7.5）：\n\n"
        "1. 【内容相关性】图片是否展示了搜索关键词对应的旅游景点/场景全貌？\n"
        "   - 10分：完美展示目标景点的标志性全景\n"
        "   - 5分：勉强相关但不是核心场景\n"
        "   - 1分：完全无关\n\n"
        "2. 【图片类型】是否为真实高清摄影照片？\n"
        "   - 10分：真实高质量摄影作品\n"
        "   - 5分：普通手机拍摄照片\n"
        "   - 1-2分：插图、截图、地图、表格、证件照、Logo、广告图、手绘等非实拍内容\n\n"
        "3. 【构图与美感】画面构图、光影、色彩是否适合社交媒体发布？\n"
        "   - 10分：构图精美，光影出色，有视觉冲击力\n"
        "   - 5分：构图平庸但可接受\n"
        "   - 1-2分：构图混乱、画面模糊、过曝/欠曝\n\n"
        "4. 【场景完整性】是否展示了有意义的完整场景？\n"
        "   - 10分：展示景点完整风貌，有空间感和纵深\n"
        "   - 5分：展示了部分场景\n"
        "   - 1-2分：只是某个器物/局部特写/文字特写/无意义的细节，看不出是什么地方\n\n"
        "5. 【干扰元素】是否有水印、文字覆盖、广告、拼图边框等干扰？\n"
        "   - 10分：画面干净无干扰\n"
        "   - 5分：有小水印但不影响整体\n"
        "   - 1-2分：大面积水印、文字覆盖、广告横幅\n\n"
        "6. 【发布价值】综合来看，这张图是否值得旅游博主在小红书上发布？\n"
        "   - 10分：非常值得，能吸引点赞收藏\n"
        "   - 5分：凑合能用\n"
        "   - 1-2分：完全不适合发布\n\n"
        "请先逐一给出6个维度的分数，然后计算加权总分（按顺序权重为：0.20, 0.15, 0.15, 0.20, 0.15, 0.15）。\n"
        "最后一行只输出一个数字作为最终总分（1.0到10.0，保留一位小数）。"
    )

    payload = {
        "model": ZHIPU_VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(ZHIPU_API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"      VLM限速(429)，等待{wait}秒后重试...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            # 提取最后一行中的数字（最终总分）
            last_line = text.strip().split("\n")[-1]
            match = re.search(r"(\d+\.?\d*)", last_line)
            if match:
                score = float(match.group(1))
                return max(1.0, min(10.0, score))
            # 若最后一行无数字，尝试从全文提取最后一个数字
            all_numbers = re.findall(r"(\d+\.?\d*)", text)
            if all_numbers:
                score = float(all_numbers[-1])
                return max(1.0, min(10.0, score))
            return 5.0
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"      VLM评估失败: {e}")
            return 5.0

    return 5.0


def _brave_image_search(query: str, count: int = 15) -> list[dict]:
    """调用Brave Image Search API，带重试"""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": count,
        "safesearch": "off",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(BRAVE_IMAGE_SEARCH_URL, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"    Brave API 错误 (尝试 {attempt+1}/{max_retries}): {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"    Brave API 请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

    return []


def _download_and_validate(img_url: str, save_path: Path) -> str | None:
    """下载图片并验证基础尺寸/大小，返回实际保存路径或None"""
    try:
        img_resp = requests.get(img_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        img_resp.raise_for_status()

        content = img_resp.content
        if len(content) < _MIN_FILE_SIZE:
            return None

        img = Image.open(io.BytesIO(content))
        w, h = img.size
        if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
            return None

        ct = img_resp.headers.get("content-type", "image/jpeg")
        if "png" in ct:
            ext = ".png"
        elif "webp" in ct:
            ext = ".webp"
        else:
            ext = ".jpg"

        final_path = save_path.with_suffix(ext)
        final_path.write_bytes(content)
        return str(final_path)

    except Exception:
        return None


def _process_query(qi: int, query: str, task_dir: Path) -> tuple[int, list[str]]:
    """处理单条搜索：Brave搜索 → 下载候选 → VLM评分 → 返回最佳图片列表"""
    print(f"  🔎 [{qi+1}] 搜索: {query}")
    results = _brave_image_search(query)

    if not results:
        print(f"    ⚠️ [{qi+1}] 未找到结果")
        return qi, []

    # 阶段1: 下载候选图片（基础尺寸/大小过滤）
    candidates = []
    for idx, item in enumerate(results):
        if len(candidates) >= _CANDIDATES_PER_QUERY:
            break
        img_url = item.get("properties", {}).get("url") or item.get("thumbnail", {}).get("src", "")
        if not img_url:
            continue
        save_path = task_dir / f"ref_q{qi}_c{idx}.jpg"
        actual_path = _download_and_validate(img_url, save_path)
        if actual_path:
            candidates.append(actual_path)

    if not candidates:
        print(f"    ⚠️ [{qi+1}] 无有效候选图")
        return qi, []

    # 阶段2: VLM 评估（通过信号量限制并发数）
    print(f"    🤖 [{qi+1}] VLM评估 {len(candidates)} 张候选图...")
    scored = []
    for cpath in candidates:
        with _VLM_SEMAPHORE:
            score = _vlm_evaluate_image(cpath, query)
        scored.append((cpath, score))
        print(f"      [{qi+1}] {Path(cpath).name}: {score:.1f}/10")

    # 按分数排序，保留最好的几张
    scored.sort(key=lambda x: x[1], reverse=True)
    group_paths = []
    for cpath, score in scored[:_KEEP_PER_QUERY]:
        if score >= 5.0:
            group_paths.append(cpath)
            print(f"    ✅ [{qi+1}] 保留: {Path(cpath).name} (VLM:{score:.1f})")

    # 清理未保留的候选文件
    kept_set = set(group_paths)
    for cpath, _ in scored:
        if cpath not in kept_set:
            try:
                Path(cpath).unlink()
            except OSError:
                pass

    return qi, group_paths


def search_images(state: dict) -> dict:
    """LangGraph节点：6组搜索并发执行，VLM评估限速保护"""
    search_queries = state.get("search_queries", [])
    image_prompts = state.get("image_prompts", [])
    location = state.get("location", state["topic"])
    topic = state["topic"]

    # 兼容旧流程：如果没有search_queries，自动生成
    if not search_queries:
        if image_prompts:
            search_queries = [f"{location} {topic} 旅游景点 实拍"] * len(image_prompts)
        else:
            search_queries = [f"{location} {topic} 旅游景点 实拍"]

    print(f"  🔎 共 {len(search_queries)} 条搜索查询，并发执行...")

    task_dir = REF_IMAGE_DIR / f"{int(time.time())}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # ── 并发执行所有搜索任务 ──
    # max_workers=3: 搜索和下载可并发，VLM由信号量(2)额外控制
    results_by_qi = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_process_query, qi, query, task_dir): qi
            for qi, query in enumerate(search_queries)
        }
        for future in as_completed(futures):
            qi = futures[future]
            try:
                _, group_paths = future.result()
                results_by_qi[qi] = group_paths
            except Exception as e:
                print(f"  ⚠️ 搜索任务 [{qi+1}] 异常: {e}")
                results_by_qi[qi] = []

    # ── 按原始顺序组装结果 ──
    ref_image_groups = []
    best_originals = []
    ref_for_gen = []
    all_ref_paths = []

    for qi in range(len(search_queries)):
        group_paths = results_by_qi.get(qi, [])
        ref_image_groups.append(group_paths)
        all_ref_paths.extend(group_paths)

        # 前2张用于发布，第3张用于img2img参考
        for gp in group_paths[:2]:
            best_originals.append(gp)
        ref_for_gen.append(
            group_paths[2] if len(group_paths) >= 3 else
            (group_paths[0] if group_paths else "")
        )

    if not all_ref_paths:
        raise RuntimeError("所有参考图片下载均失败或未通过VLM质量评估")

    print(f"  ✅ 共保留 {len(all_ref_paths)} 张高质量参考图，分 {len(ref_image_groups)} 组")
    print(f"     发布用: {len(best_originals)} 张, img2img参考: {len(ref_for_gen)} 张")
    return {
        "ref_image_paths": all_ref_paths,
        "ref_image_groups": ref_image_groups,
        "best_originals": best_originals,
        "ref_for_gen": ref_for_gen,
    }
