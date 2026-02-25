"""
自动发布小红书 — 主入口
用法：
  python main.py --topic "杭州西湖" --engine jimeng
  python main.py --topic "杭州西湖" --engine jimeng --dry-run  (不点发布)
  python main.py --resume                                      (从上次失败处继续)
  python main.py --resume --dry-run                            (继续+试运行)
"""
import argparse
import json
import sys
from pathlib import Path
from config import OUTPUT_DIR

# Fix emoji printing on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

STATE_FILE = OUTPUT_DIR / "last_state.json"


def _save_state(state: dict):
    """保存流水线状态到文件"""
    serializable = {}
    for k, v in state.items():
        # 跳过不可序列化的字段
        try:
            json.dumps(v, ensure_ascii=False)
            serializable[k] = v
        except (TypeError, ValueError):
            pass
    STATE_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state() -> dict:
    """加载上次保存的流水线状态"""
    if not STATE_FILE.exists():
        print(f"❌ 未找到已保存的状态文件: {STATE_FILE}")
        print("   请先完整运行一次流水线。")
        sys.exit(1)
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return state


def _determine_resume_point(state: dict) -> str:
    """根据已保存的状态判断应从哪一步恢复"""
    if not state.get("research_context"):
        return "research_destination"
    if not state.get("title"):
        return "generate_content"
    if not state.get("ref_image_paths"):
        return "search_images"
    if not state.get("generated_image_paths"):
        return "img2img"
    if not state.get("final_image_paths"):
        return "assemble_gallery"
    if not state.get("publish_success"):
        return "publish"
    return "done"


def _run_full_pipeline(args):
    """完整运行流水线，每步完成后保存状态"""
    from graph import build_graph

    workflow = build_graph()

    initial_state = {
        "topic": args.topic,
        "engine": args.engine,
        "dry_run": args.dry_run,
    }

    if args.skip_research:
        initial_state["research_context"] = "skip"

    print("\n  流水线步骤:")
    print("  1. 目的地调研 → 2. 生成文案 →┬→ 3a. 精简文案 ──────────→ 6. 组装画廊 → 7. 发布")
    print("                               └→ 3b. 搜索图片 → 5. 图生图 ┘")
    print()

    # 使用 stream 逐节点执行，每步保存状态
    accumulated_state = dict(initial_state)

    for event in workflow.stream(initial_state, stream_mode="updates"):
        # event 格式: {node_name: {updated_keys...}}
        for node_name, updates in event.items():
            if updates and isinstance(updates, dict):
                accumulated_state.update(updates)
                _save_state(accumulated_state)
                print(f"  💾 [{node_name}] 完成，状态已保存")

    return accumulated_state


def _run_resume(args):
    """从上次失败处恢复执行"""
    state = _load_state()
    state["dry_run"] = args.dry_run

    # 允许覆盖engine
    if args.engine:
        state["engine"] = args.engine

    resume_point = _determine_resume_point(state)

    if resume_point == "done":
        print("  ✅ 上次流水线已全部完成，无需恢复")
        print(f"     如需重新发布，使用: python main.py --resume --dry-run")
        # 即使全部完成，如果用户传了 --resume 可能想重新发布
        if args.dry_run or True:  # 允许重发
            resume_point = "publish"

    print(f"  📂 已加载状态: 标题={state.get('title', 'N/A')}")
    print(f"  ▶️ 从 [{resume_point}] 恢复执行")
    print()

    # 按顺序定义恢复步骤
    # 每步完成后保存状态，失败时已有前面的进度
    steps = _get_remaining_steps(resume_point, state.get("engine", "jimeng"))

    for step_name, step_func in steps:
        print(f"  🔄 执行: {step_name}...")
        try:
            result = step_func(state)
            if result and isinstance(result, dict):
                state.update(result)
            _save_state(state)
            print(f"  💾 [{step_name}] 完成，状态已保存")
        except Exception as e:
            _save_state(state)  # 保存到出错前的进度
            print(f"\n  ❌ [{step_name}] 失败: {e}")
            print(f"  💾 进度已保存，可用 --resume 从此步继续")
            raise

    return state


def _get_remaining_steps(resume_point: str, engine: str) -> list:
    """返回从resume_point开始需要执行的步骤列表"""
    from nodes.research_node import research_destination
    from nodes.content_node import generate_content, condense_content
    from nodes.image_search import search_images
    from nodes.image_jimeng import generate_image_jimeng
    from nodes.image_zimage import generate_image_zimage
    from nodes.assemble_node import assemble_gallery
    from nodes.publisher import publish_to_xiaohongshu

    img2img_func = generate_image_jimeng if engine == "jimeng" else generate_image_zimage

    # 完整步骤顺序（恢复时串行执行，不做并行，确保稳定）
    all_steps = [
        ("research_destination", research_destination),
        ("generate_content", generate_content),
        ("condense_content", condense_content),
        ("search_images", search_images),
        ("img2img", img2img_func),
        ("assemble_gallery", assemble_gallery),
        ("publish", publish_to_xiaohongshu),
    ]

    # 找到恢复起点
    step_names = [s[0] for s in all_steps]
    if resume_point in step_names:
        start_idx = step_names.index(resume_point)
    elif resume_point == "done":
        return []
    else:
        start_idx = 0

    return all_steps[start_idx:]


def main():
    parser = argparse.ArgumentParser(description="自动发布小红书旅游笔记")
    parser.add_argument("--topic", type=str, help="旅游主题，如：杭州西湖")
    parser.add_argument("--engine", type=str, choices=["jimeng", "zimage"], default="jimeng",
                        help="图生图引擎: jimeng(即梦4.0) 或 zimage(Z-Image本地)")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式，不点击发布按钮")
    parser.add_argument("--resume", action="store_true",
                        help="从上次失败处继续执行（加载已保存的状态）")
    parser.add_argument("--skip-research", action="store_true",
                        help="跳过目的地调研，直接用LLM生成文案")
    args = parser.parse_args()

    print("=" * 50)

    if args.resume:
        print(f"📮 恢复执行上次的流水线")
        print(f"   模式: {'试运行' if args.dry_run else '正式发布'}")
        print("=" * 50)
        state = _run_resume(args)
    else:
        if not args.topic:
            parser.error("完整运行时 --topic 为必填参数 (或使用 --resume 恢复)")

        print(f"🚀 自动发布小红书")
        print(f"   主题: {args.topic}")
        print(f"   引擎: {args.engine}")
        print(f"   模式: {'试运行' if args.dry_run else '正式发布'}")
        print("=" * 50)
        state = _run_full_pipeline(args)

    print("\n" + "=" * 50)
    if state.get("publish_success"):
        print("✅ 全流程完成！笔记已发布。")
    else:
        print("⚠️ 流程完成，但发布可能未成功，请检查。")

    print(f"   标题: {state.get('title', 'N/A')}")
    print(f"   地点: {state.get('location', 'N/A')}")
    print(f"   生成图片: {len(state.get('generated_image_paths', []))}张")
    print(f"   最终发布: {len(state.get('final_image_paths', state.get('generated_image_paths', [])))}张")
    print(f"   💡 如需重新发布: python main.py --resume")
    print("=" * 50)


if __name__ == "__main__":
    main()
