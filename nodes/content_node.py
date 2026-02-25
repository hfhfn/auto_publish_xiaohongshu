"""
文案生成节点 — 使用智谱GLM生成小红书风格旅游文案
通过 OpenAI 兼容接口调用，支持智谱 → DeepSeek 自动降级
流式输出实时显示生成进度，彻底避免超时问题
"""
import re
import sys
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import (
    ZHIPU_API_KEY, ZHIPU_MODEL, ZHIPU_API_URL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
)


class XHSNote(BaseModel):
    """小红书笔记结构"""
    title: str = Field(description="标题，20字以内，吸引眼球")
    content: str = Field(
        description="正文内容，800-1000字（严格不超过1000字），小红书种草风格，口语化、活泼、有感染力，适当使用emoji。"
                    "必须涵盖所有重要景点并逐一介绍亮点，包含美食推荐、交通指南、住宿建议，"
                    "按合理游览路线组织内容，信息密度要高，用精炼的语言覆盖尽可能多的内容"
    )
    tags: list[str] = Field(description="5个相关标签")
    location: str = Field(description="具体地点名称，如：杭州西湖风景区")
    image_prompts: list[str] = Field(
        description="6个英文图片描述，用于AI生图，要求写实摄影风格。"
                    "应覆盖：主要景点（至少4个不同景点）、当地美食或特色小吃、"
                    "风土人情或街景建筑等，确保每个prompt对应不同的场景"
    )
    search_queries: list[str] = Field(
        description="与image_prompts一一对应的6个中文搜索关键词，"
                    "用于图片搜索引擎搜索真实照片，包含具体地名+场景关键词，"
                    "每条搜索词应覆盖不同的景色和角度以确保多样性，"
                    "例如：'壶口瀑布 黄河 高清摄影'"
    )
    collection_name: str = Field(
        description="合集名称，20字以内，使用通用旅行主题，便于同一目的地的多篇笔记复用，"
                    "例如：'山西旅行记录'、'江浙沪周末游'，不要加入'深度游'、'攻略'等具体修饰词"
    )
    collection_desc: str = Field(
        description="合集简介，50字以内，概括合集内容"
    )


# 用 JsonOutputParser 生成格式说明，嵌入 prompt 作为双保险
_parser = JsonOutputParser(pydantic_object=XHSNote)

SYSTEM_PROMPT = """你是一个专业的小红书旅游博主，擅长写种草文案。
请根据用户给出的旅游主题，生成一篇小红书笔记内容。

要求：
1. 文案风格：口语化、活泼、有感染力，适当使用emoji
2. 正文内容必须800-1000字（严格不超过1000字，这是平台限制），信息密度要高，用精炼语言覆盖所有景点
3. image_prompts 用英文描述，写实摄影风格，6个prompt应覆盖：至少4个不同景点、当地美食、风土人情/街景
4. search_queries 必须与 image_prompts 一一对应，每个是中文搜索关键词，包含具体地名和场景关键词，适合在图片搜索引擎中搜索到高质量真实照片。
   搜索词中必须包含准确的目的地名称，不要出现其他城市或省份的名称，确保搜索结果是目标地点的图片
5. collection_name 使用通用旅行主题名，便于多篇笔记复用同一合集（如"山西旅行记录"），不要加"深度游""攻略"等修饰词
6. collection_desc 是合集简介，50字以内
7. 如果提供了目的地调研资料，你必须把调研资料中提到的所有景点、美食、交通、住宿信息都融入文案中：
   - 每个景点都要提到并介绍其亮点特色，不能遗漏
   - 按合理的地理位置或游览顺序组织路线
   - 包含具体的美食推荐（菜名、店名）
   - 包含交通方式和住宿建议
   - 对调研资料进行润色和总结，不是删减，是扩写和美化
   - image_prompts 和 search_queries 也要覆盖调研资料中提到的不同景点，不要只选最知名的

""" + _parser.get_format_instructions().replace("{", "{{").replace("}", "}}")


# ── LLM 提供者配置（按优先级排列）──────────────────────────────
_LLM_PROVIDERS = [
    {
        "name": "智谱AI",
        "model": ZHIPU_MODEL,
        "api_key": ZHIPU_API_KEY,
        "base_url": ZHIPU_API_URL.replace("/chat/completions", ""),
        "timeout": 60,   # 智谱给60秒，超时就降级
    },
    {
        "name": "DeepSeek",
        "model": DEEPSEEK_MODEL,
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "timeout": 60,  # 最后兜底
    },
]


def _fix_json_escapes(text: str) -> str:
    """修复LLM生成的JSON中的非法转义符。

    LLM有时会输出非标准转义（如反斜杠后跟中文），需要将孤立的反斜杠转为双反斜杠。
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def _unwrap_json(data: dict) -> dict:
    """解包可能被嵌套的 JSON 响应。

    部分模型（如智谱）会将结果包在 {"answer": {...}} 或其他单 key 中。
    如果顶层只有一个 key 且其值是 dict，就解包取出内层；
    否则认为是标准的直接输出，原样返回。
    """
    if len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, dict):
            return only_value
    return data


def _condense_if_needed(title: str, content: str, llm) -> tuple[str, str]:
    """检查标题和正文是否超限，超限则调用LLM重新生成/精简。
    最终保证标题≤20字、正文≤1000字（兜底硬截断）。
    """
    # 标题：20字以内，超出则让LLM重新生成
    if len(title) > 20:
        print(f"  ✂️ 标题 {len(title)} 字超出20字限制，正在让LLM重写...")
        try:
            resp = llm.invoke(
                f"以下小红书笔记标题超出了20字限制（当前{len(title)}字）：\n"
                f"「{title}」\n\n"
                "请重写一个不超过20字的标题，要求：\n"
                "1. 严格不超过20个字符（含标点和emoji）\n"
                "2. 保持吸引眼球的种草风格\n"
                "3. 保留核心主题信息\n"
                "直接输出新标题，不要加引号或任何说明。"
            )
            new_title = resp.content.strip().strip("「」\"'""''")
            if 0 < len(new_title) <= 20:
                title = new_title
                print(f"  ✅ 标题重写完成: {title}（{len(title)}字）")
            else:
                title = title[:20]
                print(f"  ⚠️ 重写结果仍超限，强制截取: {title}")
        except Exception as e:
            title = title[:20]
            print(f"  ⚠️ 标题重写失败: {e}，强制截取")

    # 正文：1000字以内，超出则让LLM精简
    if len(content) > 1000:
        print(f"  ✂️ 正文 {len(content)} 字超出1000字限制，正在让LLM精简...")
        try:
            compress_resp = llm.invoke(
                f"请将以下小红书旅游文案精简到950-1000字以内（当前{len(content)}字）。\n"
                "要求：保留所有景点和关键信息，保持口语化种草风格和emoji，"
                "通过精炼语言来缩短，不要删除整段内容。\n"
                "直接输出精简后的文案，不要加任何说明文字。\n\n"
                f"{content}"
            )
            condensed = compress_resp.content.strip()
            if 500 < len(condensed) <= 1000:
                content = condensed
                print(f"  ✅ 精简完成: {len(content)} 字")
            else:
                print(f"  ⚠️ 精简结果 {len(condensed)} 字不在合理范围")
        except Exception as e:
            print(f"  ⚠️ LLM精简失败: {e}")

    # 兜底硬截断：确保绝不超限
    if len(title) > 20:
        title = title[:20]
        print(f"  🔒 标题兜底截取至20字")
    if len(content) > 1000:
        # 在最后一个完整句子处截断，避免半句话
        cut = content[:1000]
        # 从末尾往前找句号/感叹号/换行作为断句点
        for sep in ["。", "！", "!", "\n", "～", "~", "；"]:
            last_pos = cut.rfind(sep)
            if last_pos > 800:
                cut = cut[:last_pos + 1]
                break
        content = cut
        print(f"  🔒 正文兜底截断至{len(content)}字")

    return title, content


def _stream_generate(chain, invoke_params: dict, provider_name: str) -> str:
    """流式调用 chain，实时打印进度，返回完整文本。失败时抛出异常。"""
    chunks = []
    char_count = 0
    last_reported = 0

    for chunk in chain.stream(invoke_params):
        text = chunk.content if hasattr(chunk, "content") else ""
        if text:
            chunks.append(text)
            char_count += len(text)
            if char_count - last_reported >= 200:
                print(f"\r  ⏳ [{provider_name}] 已生成 {char_count} 字符...", end="", flush=True)
                last_reported = char_count

    print(f"\r  ⏳ [{provider_name}] 已生成 {char_count} 字符 ✔️   ")
    sys.stdout.flush()

    if char_count == 0:
        raise RuntimeError("模型返回空内容")

    return "".join(chunks)


def generate_content(state: dict) -> dict:
    """LangGraph节点：根据主题生成小红书文案（流式输出 + 自动降级）"""
    topic = state["topic"]
    research_context = state.get("research_context", "")
    if research_context == "skip":
        research_context = ""

    # 构建 prompt（所有 provider 共用）
    if research_context:
        user_message = (
            "请为以下旅游主题生成小红书笔记：{topic}\n\n"
            "以下是目的地调研资料，请务必将其中提到的所有景点、美食、交通、住宿等信息"
            "全部融入文案，不要遗漏任何景点。对资料进行润色和扩写，而不是删减：\n\n"
            "{research_context}"
        )
    else:
        user_message = "请为以下旅游主题生成小红书笔记：{topic}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", user_message),
    ])
    invoke_params = {"topic": topic, "research_context": research_context}

    # 按优先级尝试每个 provider
    last_error = None
    for provider in _LLM_PROVIDERS:
        name = provider["name"]
        print(f"  🎬 正在使用{name}生成文案: {topic}...")

        llm = ChatOpenAI(
            model=provider["model"],
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            temperature=0.8,
            timeout=provider["timeout"],
            streaming=True,
        )
        llm_json = llm.bind(response_format={"type": "json_object"})
        chain = prompt | llm_json

        try:
            full_text = _stream_generate(chain, invoke_params, name)
            try:
                data = json.loads(full_text)
            except json.JSONDecodeError:
                data = json.loads(_fix_json_escapes(full_text))
            data = _unwrap_json(data)
            result = XHSNote(**data)

            print(f"  ✅ 文案生成成功 [{name}]: {result.title}")
            print(f"  📏 原始长度: 标题={len(result.title)}字(限20), 正文={len(result.content)}字(限1000)")

            return {
                "title": result.title,
                "content": result.content,
                "tags": result.tags,
                "location": result.location,
                "image_prompts": result.image_prompts,
                "search_queries": result.search_queries,
                "collection_name": result.collection_name,
                "collection_desc": result.collection_desc,
            }
        except Exception as e:
            last_error = e
            print(f"\n  ⚠️ {name}调用失败: {e}")
            print(f"  🔄 尝试降级到下一个模型...")

    # 所有 provider 都失败
    raise RuntimeError(f"所有LLM提供者均失败，最后一个错误: {last_error}")


def condense_content(state: dict) -> dict:
    """LangGraph节点：校验并精简标题和正文长度（与图片搜索并行执行）"""
    title = state.get("title", "")
    content = state.get("content", "")

    print(f"  ✂️ 文案校验精简: 标题={len(title)}字(限20), 正文={len(content)}字(限1000)")

    # 如果都不超限，直接通过
    if len(title) <= 20 and len(content) <= 1000:
        print(f"  ✅ 长度校验通过，无需精简")
        return {}

    # 需要精简，创建 LLM 实例
    provider = _LLM_PROVIDERS[0]
    llm = ChatOpenAI(
        model=provider["model"],
        api_key=provider["api_key"],
        base_url=provider["base_url"],
        temperature=0.3,
        timeout=provider["timeout"],
    )

    title, content = _condense_if_needed(title, content, llm)

    if len(title) <= 20 and len(content) <= 1000:
        print(f"  ✅ 精简完成: 标题={len(title)}字, 正文={len(content)}字")
    else:
        print(f"  ⚠️ 精简后仍超限: 标题={len(title)}字, 正文={len(content)}字")

    return {"title": title, "content": content}
