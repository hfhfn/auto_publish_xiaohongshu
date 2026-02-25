"""
图片搜索节点 — 使用 Brave + Tavily 双引擎按prompt分组搜索真实景点图片并下载，
通过 GLM VLM 进行图片质量评估。搜索并发执行，VLM调用限速保护。
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from config import (
    BRAVE_API_KEY, BRAVE_IMAGE_SEARCH_URL, REF_IMAGE_DIR,
    ZHIPU_API_KEY, ZHIPU_VLM_MODEL, ZHIPU_API_URL,
    TAVILY_API_KEY,
)

# 最小图片尺寸和文件大小（基础预过滤，避免对明显无用的图片调VLM浪费token）
_MIN_DIMENSION = 400
_MIN_FILE_SIZE = 50 * 1024  # 50KB

# 每条query下载的候选数（Brave+Tavily合并后，VLM评估保留前3）
_CANDIDATES_PER_QUERY = 12
# 每条query最终保留: 前2张发布 + 第3张做img2img参考
_KEEP_PER_QUERY = 3
# 同一域名最多取几张候选图（强制来源多样性）
_MAX_PER_DOMAIN = 2

# VLM并发控制：最多2个同时调用，避免API限速
_VLM_SEMAPHORE = threading.Semaphore(2)

# ── 图片来源域名过滤 ──────────────────────────────────────
# 黑名单：水印重、新闻站点、低质量图库（这些站点的图几乎都带大面积水印或Logo）
_DOMAIN_BLACKLIST = {
    "xinhuanet.com", "news.cn", "people.com.cn", "people.cn",
    "chinanews.com", "chinanews.com.cn", "cctv.com", "cctv.cn",
    "china.com.cn", "china.com", "gmw.cn", "cnr.cn",
    "youth.cn", "cyol.com", "ce.cn", "huanqiu.com",
    "thepaper.cn", "163.com", "sina.com.cn", "sohu.com",
    "qq.com", "ifeng.com", "hexun.com", "caixin.com",
    "bjnews.com.cn", "chinadaily.com.cn",
    # 低质量图库/百科缩略图
    "bkimg.cdn.bcebos.com", "so.com", "sogou.com",
}

# 优质来源：旅游网站、摄影社区、地方文旅（搜索结果中优先保留）
_PREFERRED_DOMAINS = {
    "mafengwo.cn", "mafengwo.com",           # 马蜂窝
    "ctrip.com", "dianping.com",              # 携程、大众点评
    "qunar.com", "tuniu.com", "fliggy.com",   # 去哪儿、途牛、飞猪
    "poco.cn", "tuchong.com", "500px.com",    # 摄影社区
    "zcool.com.cn", "lofter.com",             # 设计/摄影
    "unsplash.com", "pexels.com",             # 免费高质量图库
    "flickr.com", "shutterstock.com",
    "dpfile.com",                              # 点评图片CDN
    "youimg1.c-ctrip.com",                    # 携程图片CDN
}


def _extract_domain(url: str) -> str:
    """从URL中提取主域名（去掉www等前缀，保留核心域名）"""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        # 去掉 www. / m. / img. 等前缀，取最后两段作为主域名
        parts = host.lower().split(".")
        if len(parts) >= 2:
            # 处理 .com.cn / .org.cn 等双后缀
            if parts[-1] in ("cn", "com", "net", "org") and parts[-2] in ("com", "org", "net", "gov", "edu"):
                return ".".join(parts[-3:]) if len(parts) >= 3 else host
            return ".".join(parts[-2:])
        return host
    except Exception:
        return ""


def _is_blacklisted(source_domain: str) -> bool:
    """检查域名是否在黑名单中"""
    source_lower = source_domain.lower()
    for blocked in _DOMAIN_BLACKLIST:
        if blocked in source_lower:
            return True
    return False


def _is_preferred(source_domain: str) -> bool:
    """检查域名是否为优质来源"""
    source_lower = source_domain.lower()
    for preferred in _PREFERRED_DOMAINS:
        if preferred in source_lower:
            return True
    return False


# ── LLM / SDK 客户端 ──────────────────────────────────────
_vlm = ChatOpenAI(
    model=ZHIPU_VLM_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_API_URL.replace("/chat/completions", ""),
    temperature=0.1,
    max_tokens=300,
    timeout=30,
)

_tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def _vlm_evaluate_image(image_path: str, query: str, topic: str = "") -> float:
    """调用 GLM VLM 评估图片质量，返回1.0-10.0分（含0.5小数）。
    内置429限速重试。
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")

    location_warning = ""
    if topic:
        location_warning = (
            f'\n⚠️ 特别注意：本次搜索的目标地点是"{topic}"。'
            f'如果图片中出现了与"{topic}"无关的其他城市名、省份名或地名文字'
            '（如图片上叠加了其他城市的名字），"地点一致性"必须判为1分，'
            '这会严重拉低总分。\n'
        )

    prompt = (
        f'请严格评估这张图片作为旅游博主发布素材的质量。搜索关键词："{query}"'
        f'{location_warning}\n\n'
        "请从以下7个维度逐一打分，每个维度1-10分，允许0.5的小数（如7.5）：\n\n"
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
        "6. 【地点一致性】图片中是否出现了与目标地点不符的文字？\n"
        "   - 10分：没有任何地名文字，或文字与目标地点一致\n"
        "   - 5分：有少量不确定的文字\n"
        "   - 1分：图片上明显出现了其他城市/省份的名字（如目标是商丘但图上写着武汉）\n\n"
        "7. 【发布价值】综合来看，这张图是否值得旅游博主在小红书上发布？\n"
        "   - 10分：非常值得，能吸引点赞收藏\n"
        "   - 5分：凑合能用\n"
        "   - 1-2分：完全不适合发布\n\n"
        "请先逐一给出7个维度的分数，然后计算加权总分（按顺序权重为：0.15, 0.10, 0.10, 0.15, 0.10, 0.25, 0.15）。\n"
        "注意：如果'地点一致性'为1分，无论其他维度多高，总分不应超过3.0。\n"
        "最后一行只输出一个数字作为最终总分（1.0到10.0，保留一位小数）。"
    )

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ])

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _vlm.invoke([message])
            text = response.content.strip()
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
            # 处理429限速等异常
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = 3 * (attempt + 1)
                print(f"      VLM限速(429)，等待{wait}秒后重试...")
                time.sleep(wait)
                continue
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


def _tavily_image_search(query: str, max_results: int = 10) -> list[str]:
    """调用Tavily Search API获取图片URL列表，带重试。
    返回去重后的图片URL字符串列表。
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = _tavily_client.search(
                query,
                include_images=True,
                max_results=max_results,
                search_depth="advanced",
            )
            images = result.get("images", [])
            # images 可能是字符串列表 ["url1", "url2"] 或字典列表 [{"url": "..."}]
            urls = []
            for img in images:
                if isinstance(img, str):
                    urls.append(img)
                elif isinstance(img, dict):
                    url = img.get("url", "")
                    if url:
                        urls.append(url)
            return urls
        except Exception as e:
            print(f"    Tavily API 请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                # Tavily失败不阻断流程，返回空列表
                print(f"    ⚠️ Tavily搜索最终失败，仅使用Brave结果")
                return []

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


def _process_query(qi: int, query: str, task_dir: Path, topic: str = "") -> tuple[int, list[tuple[str, float]]]:
    """处理单条搜索：Brave 1次 + Tavily 1次 → 域名过滤 → 下载候选 → VLM评分
    返回 (qi, scored_passing) — 通过阈值的 (path, score)，按分数降序，最多 _KEEP_PER_QUERY 张。
    """
    print(f"  🔎 [{qi+1}] 搜索: {query}")

    # ── 每个query只搜2次：Brave 1次 + Tavily 1次 ──
    brave_results_raw = _brave_image_search(query, count=20)
    tavily_urls = _tavily_image_search(query, max_results=15)

    # ── 合并去重 ──
    seen_urls = set()
    brave_results = []
    for item in brave_results_raw:
        img_url = item.get("properties", {}).get("url") or item.get("thumbnail", {}).get("src", "")
        if img_url and img_url not in seen_urls:
            seen_urls.add(img_url)
            brave_results.append(item)

    tavily_results = []
    for url in tavily_urls:
        if url and url not in seen_urls:
            seen_urls.add(url)
            tavily_results.append({"properties": {"url": url}, "source": "", "_from_tavily": True})

    print(f"    📡 [{qi+1}] Brave: {len(brave_results)}条, Tavily: {len(tavily_results)}条")

    all_results = brave_results + tavily_results
    if not all_results:
        print(f"    ⚠️ [{qi+1}] 未找到结果")
        return qi, []

    # ── 域名过滤 + 多样性控制 + 下载候选图片 ──
    candidates = []
    domain_count = {}
    skipped_blacklist = 0
    skipped_domain_limit = 0

    for idx, item in enumerate(all_results):
        if len(candidates) >= _CANDIDATES_PER_QUERY:
            break

        img_url = item.get("properties", {}).get("url") or item.get("thumbnail", {}).get("src", "")
        if not img_url:
            continue

        source = item.get("source", "")
        domain = _extract_domain(source) if source else _extract_domain(img_url)

        if _is_blacklisted(domain):
            skipped_blacklist += 1
            continue

        if domain_count.get(domain, 0) >= _MAX_PER_DOMAIN:
            skipped_domain_limit += 1
            continue

        tag = "T" if item.get("_from_tavily") else "B"
        save_path = task_dir / f"ref_q{qi}_{tag}{idx}.jpg"
        actual_path = _download_and_validate(img_url, save_path)
        if actual_path:
            candidates.append((actual_path, domain))
            domain_count[domain] = domain_count.get(domain, 0) + 1

    if skipped_blacklist > 0 or skipped_domain_limit > 0:
        print(f"    🚫 [{qi+1}] 过滤: {skipped_blacklist}张黑名单, {skipped_domain_limit}张重复来源")

    if candidates:
        domains_used = set(d for _, d in candidates)
        print(f"    📊 [{qi+1}] {len(candidates)}张候选, {len(domains_used)}个来源: {', '.join(list(domains_used)[:5])}")

    if not candidates:
        print(f"    ⚠️ [{qi+1}] 无有效候选图")
        return qi, []

    # ── VLM 评估 ──
    print(f"    🤖 [{qi+1}] VLM评估 {len(candidates)} 张候选图...")
    scored = []
    for cpath, domain in candidates:
        with _VLM_SEMAPHORE:
            score = _vlm_evaluate_image(cpath, query, topic)
        if _is_preferred(domain):
            score = min(10.0, score + 0.3)
        scored.append((cpath, score))
        print(f"      [{qi+1}] {Path(cpath).name}: {score:.1f}/10 ({domain})")

    # 按分数降序，取通过阈值的前 _KEEP_PER_QUERY 张
    scored.sort(key=lambda x: x[1], reverse=True)
    passing = [(p, s) for p, s in scored[:_KEEP_PER_QUERY] if s >= 5.0]

    for cpath, score in passing:
        print(f"    ✅ [{qi+1}] 保留: {Path(cpath).name} (VLM:{score:.1f})")

    # 清理未保留的候选文件
    kept_set = set(p for p, _ in passing)
    for cpath, _ in scored:
        if cpath not in kept_set:
            try:
                Path(cpath).unlink()
            except OSError:
                pass

    return qi, passing


def search_images(state: dict) -> dict:
    """LangGraph节点：Brave+Tavily双引擎并发搜索，VLM评估限速保护。
    每组保留前3张: 前2张发布用原图，第3张做img2img参考。总发布图最多18张。
    """
    search_queries = state.get("search_queries", [])
    image_prompts = state.get("image_prompts", [])
    location = state.get("location", state["topic"])
    topic = state["topic"]

    if not search_queries:
        if image_prompts:
            search_queries = [f"{location} {topic} 旅游景点 实拍"] * len(image_prompts)
        else:
            search_queries = [f"{location} {topic} 旅游景点 实拍"]

    print(f"  🔎 共 {len(search_queries)} 条搜索 (Brave+Tavily各1次/条，共{len(search_queries)*2}次API调用)")

    task_dir = REF_IMAGE_DIR / f"{int(time.time())}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # ── 并发执行所有搜索任务 ──
    results_by_qi = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_process_query, qi, query, task_dir, topic): qi
            for qi, query in enumerate(search_queries)
        }
        for future in as_completed(futures):
            qi = futures[future]
            try:
                _, scored_passing = future.result()
                results_by_qi[qi] = scored_passing
            except Exception as e:
                print(f"  ⚠️ 搜索任务 [{qi+1}] 异常: {e}")
                results_by_qi[qi] = []

    # ── 按原始顺序组装结果 ──
    ref_image_groups = []
    best_originals = []
    ref_for_gen = []
    all_ref_paths = []

    for qi in range(len(search_queries)):
        scored_passing = results_by_qi.get(qi, [])
        # scored_passing 已按分数降序: [(path, score), ...]
        group_paths = [p for p, _ in scored_passing]

        ref_image_groups.append(group_paths)
        all_ref_paths.extend(group_paths)

        # 前2张（最高分）→ 发布用原图
        for gp in group_paths[:2]:
            best_originals.append(gp)

        # 第3张 → img2img生成参考（质量排第三，适合作为生图基底）
        if len(scored_passing) >= 3:
            ref_path = scored_passing[2][0]
            ref_score = scored_passing[2][1]
            ref_for_gen.append(ref_path)
            print(f"    🎯 [{qi+1}] img2img参考: {Path(ref_path).name} (VLM:{ref_score:.1f})")
        elif scored_passing:
            ref_for_gen.append(scored_passing[-1][0])
        else:
            ref_for_gen.append("")

    if not all_ref_paths:
        raise RuntimeError("所有参考图片下载均失败或未通过VLM质量评估")

    print(f"  ✅ 共保留 {len(all_ref_paths)} 张参考图 (6组×3张)，上限18张")
    print(f"     发布用: {len(best_originals)} 张, img2img参考: {len(ref_for_gen)} 张")
    return {
        "ref_image_paths": all_ref_paths,
        "ref_image_groups": ref_image_groups,
        "best_originals": best_originals,
        "ref_for_gen": ref_for_gen,
    }
