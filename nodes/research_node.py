"""
目的地调研节点 — 使用Brave Web Search搜索旅游攻略，为文案生成提供真实信息
"""
import requests
import time
from config import BRAVE_API_KEY, BRAVE_WEB_SEARCH_URL

# 搜索结果摘要的最大字符数
_MAX_CONTEXT_CHARS = 8000


def _brave_web_search(query: str, count: int = 5) -> list[dict]:
    """调用Brave Web Search API，返回搜索结果列表"""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": count,
        "extra_snippets": True,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(BRAVE_WEB_SEARCH_URL, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"    Brave Web Search 错误 (尝试 {attempt+1}/{max_retries}): {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            return data.get("web", {}).get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"    Brave Web Search 请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return []


def _extract_snippets(results: list[dict]) -> str:
    """从搜索结果中提取标题、描述和extra_snippets，拼接为文本"""
    parts = []
    for item in results:
        title = item.get("title", "")
        description = item.get("description", "")
        extra = item.get("extra_snippets", [])

        if title:
            parts.append(title)
        if description:
            parts.append(description)
        for snippet in extra:
            if snippet:
                parts.append(snippet)

    return "\n".join(parts)


def research_destination(state: dict) -> dict:
    """LangGraph节点：搜索旅游攻略，提取摘要作为文案生成的背景资料"""
    # 若已有 research_context（如 --skip-research 传入空串），直接跳过
    if state.get("research_context"):
        print("  ⏭️ 已有调研资料，跳过目的地调研")
        return {}

    topic = state["topic"]
    print(f"  🔍 正在调研目的地: {topic}...")

    queries = [
        f"{topic} 旅游攻略 必去景点 美食推荐",
        f"{topic} 自由行 交通 住宿 路线规划",
    ]

    all_snippets = []
    for qi, query in enumerate(queries):
        print(f"    [{qi+1}/{len(queries)}] 搜索: {query}")
        results = _brave_web_search(query)
        if results:
            snippet_text = _extract_snippets(results)
            all_snippets.append(snippet_text)
            print(f"    ✅ 获取 {len(results)} 条结果")
        else:
            print(f"    ⚠️ 未获取到结果")

    research_context = "\n\n".join(all_snippets)

    # 截断到最大字符数
    if len(research_context) > _MAX_CONTEXT_CHARS:
        research_context = research_context[:_MAX_CONTEXT_CHARS]
        print(f"  ✂️ 调研资料已截断到 {_MAX_CONTEXT_CHARS} 字符")

    if research_context:
        print(f"  ✅ 目的地调研完成，共 {len(research_context)} 字符")
    else:
        print(f"  ⚠️ 未获取到调研资料，将使用纯LLM生成")

    return {"research_context": research_context}
