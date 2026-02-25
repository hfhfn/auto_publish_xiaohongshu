"""
LangGraph 工作流定义 — 串联目的地调研、文案生成、图片搜索、图生图、画廊组装、发布六个节点
"""
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END

from nodes.research_node import research_destination
from nodes.content_node import generate_content
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


def route_engine(state: PublishState) -> str:
    """条件路由：根据engine选择图生图节点"""
    return state["engine"]


def build_graph() -> StateGraph:
    """构建LangGraph工作流"""
    graph = StateGraph(PublishState)

    # 添加节点
    graph.add_node("research_destination", research_destination)
    graph.add_node("generate_content", generate_content)
    graph.add_node("search_images", search_images)
    graph.add_node("img2img_jimeng", generate_image_jimeng)
    graph.add_node("img2img_zimage", generate_image_zimage)
    graph.add_node("assemble_gallery", assemble_gallery)
    graph.add_node("publish", publish_to_xiaohongshu)

    # 定义边
    graph.add_edge(START, "research_destination")
    graph.add_edge("research_destination", "generate_content")
    graph.add_edge("generate_content", "search_images")

    # 条件路由：选择图生图引擎
    graph.add_conditional_edges(
        "search_images",
        route_engine,
        {
            "jimeng": "img2img_jimeng",
            "zimage": "img2img_zimage",
        },
    )

    # 两个图生图节点都汇聚到画廊组装节点
    graph.add_edge("img2img_jimeng", "assemble_gallery")
    graph.add_edge("img2img_zimage", "assemble_gallery")

    graph.add_edge("assemble_gallery", "publish")
    graph.add_edge("publish", END)

    return graph.compile()
