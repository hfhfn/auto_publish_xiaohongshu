"""
LangGraph 工作流定义 — 串联目的地调研、文案生成、文案精简、图片搜索、图生图、画廊组装、发布
"""
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END

from nodes.research_node import research_destination
from nodes.content_node import generate_content, condense_content
from nodes.image_search import search_images
from nodes.image_jimeng import generate_image_jimeng
from nodes.image_zimage import generate_image_zimage
from nodes.assemble_node import assemble_gallery
from nodes.publisher import publish_to_xiaohongshu


class PublishState(TypedDict):
    """工作流状态"""
    # 输入配置
    topic: str                                    # 旅游主题
    engine: Literal["jimeng", "zimage"]            # 图生图引擎
    dry_run: bool                                  # 试运行模式
    # 目的地调研
    research_context: str                          # 搜索攻略摘要，供文案生成参考
    # 文案生成结果
    title: str
    content: str
    tags: list[str]
    location: str
    image_prompts: list[str]
    search_queries: list[str]                      # 与image_prompts一一对应的中文搜索词
    collection_name: str                             # 合集名称
    collection_desc: str                             # 合集简介
    # 图片路径
    ref_image_paths: list[str]                     # 搜索到的参考图（全部，兼容字段）
    ref_image_groups: list[list[str]]              # 按prompt分组的参考图路径
    best_originals: list[str]                      # 每组前2张用于发布的原图
    ref_for_gen: list[str]                         # 每组第3张用于img2img参考（不发布）
    generated_image_paths: list[str]               # 图生图生成的图片
    final_image_paths: list[str]                   # 组装后的最终发布图片
    # 发布状态
    publish_success: bool


def img2img(state: PublishState) -> dict:
    """统一图生图节点：根据engine字段分派到对应引擎"""
    engine = state.get("engine", "jimeng")
    if engine == "zimage":
        return generate_image_zimage(state)
    return generate_image_jimeng(state)


def build_graph() -> StateGraph:
    """构建LangGraph工作流

    并行流水线（condense_content 与图片分支并行，但不做 fan-in）:
        research → generate_content ─┬→ condense_content → END  (精简文案写入共享state)
                                     └→ search_images → img2img → assemble_gallery → publish → END
    condense_content 几秒即完成，图片流水线需数分钟，
    publish 执行时 condense_content 的结果早已写入 state。
    """
    graph = StateGraph(PublishState)

    # 添加节点
    graph.add_node("research_destination", research_destination)
    graph.add_node("generate_content", generate_content)
    graph.add_node("condense_content", condense_content)
    graph.add_node("search_images", search_images)
    graph.add_node("img2img", img2img)
    graph.add_node("assemble_gallery", assemble_gallery)
    graph.add_node("publish", publish_to_xiaohongshu)

    # 调研 → 文案生成
    graph.add_edge(START, "research_destination")
    graph.add_edge("research_destination", "generate_content")

    # 并行分支：文案精简 和 图片搜索同时启动
    graph.add_edge("generate_content", "condense_content")
    graph.add_edge("generate_content", "search_images")

    # 文案分支：精简完成后直接结束（结果已写入共享state，publish能读到）
    graph.add_edge("condense_content", END)

    # 图片分支：搜索 → 图生图 → 组装 → 发布（唯一通向 publish 的路径）
    graph.add_edge("search_images", "img2img")
    graph.add_edge("img2img", "assemble_gallery")
    graph.add_edge("assemble_gallery", "publish")
    graph.add_edge("publish", END)

    return graph.compile()
