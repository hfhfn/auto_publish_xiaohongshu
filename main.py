"""
自动发布小红书 — 主入口
用法：
  python main.py --topic "杭州西湖" --engine jimeng
  python main.py --topic "杭州西湖" --engine zimage
  python main.py --topic "杭州西湖" --engine jimeng --dry-run  (不点发布)
  python main.py --skip-to-publish                             (跳过生成，直接发布上次的内容)
  python main.py --skip-to-publish --dry-run                   (跳过生成，试运行发布)
"""
import argparse
import json
import sys
from pathlib import Path
from graph import build_graph
from config import OUTPUT_DIR

# Fix emoji printing on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

STATE_FILE = OUTPUT_DIR / "last_state.json"


def _save_state(state: dict):
    """保存流水线状态到文件，供 --skip-to-publish 复用"""
    serializable = {k: v for k, v in state.items() if k != "publish_success"}
    STATE_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  💾 状态已保存到 {STATE_FILE}")


def _load_state() -> dict:
    """加载上次保存的流水线状态"""
    if not STATE_FILE.exists():
        print(f"❌ 未找到已保存的状态文件: {STATE_FILE}")
        print("   请先完整运行一次流水线生成文案和图片。")
        sys.exit(1)
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    print(f"  📂 已加载状态: 标题={state.get('title', 'N/A')}, 图片={len(state.get('generated_image_paths', []))}张")
    return state


def main():
    parser = argparse.ArgumentParser(description="自动发布小红书旅游笔记")
    parser.add_argument("--topic", type=str, help="旅游主题，如：杭州西湖")
    parser.add_argument("--engine", type=str, choices=["jimeng", "zimage"], default="jimeng",
                        help="图生图引擎: jimeng(即梦4.0) 或 zimage(Z-Image本地)")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式，不点击发布按钮")
    parser.add_argument("--skip-to-publish", action="store_true",
                        help="跳过文案和图片生成，直接使用上次结果发布")
    parser.add_argument("--skip-research", action="store_true",
                        help="跳过目的地调研，直接用LLM生成文案（节省API调用）")
    args = parser.parse_args()

    if args.skip_to_publish:
        # ===== 跳过生成，直接发布 =====
        print("=" * 50)
        print("📮 跳过生成，直接发布上次生成的内容")
        print(f"   模式: {'试运行' if args.dry_run else '正式发布'}")
        print("=" * 50)

        state = _load_state()
        state["dry_run"] = args.dry_run

        from nodes.publisher import publish_to_xiaohongshu
        result = publish_to_xiaohongshu(state)
        state.update(result)
    else:
        # ===== 完整流水线 =====
        if not args.topic:
            parser.error("完整运行时 --topic 为必填参数")

        print("=" * 50)
        print(f"🚀 自动发布小红书")
        print(f"   主题: {args.topic}")
        print(f"   引擎: {args.engine}")
        print(f"   模式: {'试运行' if args.dry_run else '正式发布'}")
        print("=" * 50)

        workflow = build_graph()

        initial_state = {
            "topic": args.topic,
            "engine": args.engine,
            "dry_run": args.dry_run,
        }

        # --skip-research: 预填 research_context 使调研节点跳过
        if args.skip_research:
            initial_state["research_context"] = "skip"

        print("\n🔍 Step 1/6: 目的地调研...")
        print("📝 Step 2/6: 生成文案...")
        print("🔍 Step 3/6: 搜索真实景点图片...")
        print("🎨 Step 4/6: 图生图生成...")
        print("🖼️ Step 5/6: 组装图片画廊...")
        print("📮 Step 6/6: 发布到小红书...")
        print()

        state = workflow.invoke(initial_state)
        _save_state(state)

    print("\n" + "=" * 50)
    if state.get("publish_success"):
        print("✅ 全流程完成！笔记已发布。")
    else:
        print("⚠️ 流程完成，但发布可能未成功，请检查。")

    print(f"   标题: {state.get('title', 'N/A')}")
    print(f"   地点: {state.get('location', 'N/A')}")
    print(f"   生成图片: {len(state.get('generated_image_paths', []))}张")
    print(f"   最终发布: {len(state.get('final_image_paths', state.get('generated_image_paths', [])))}张")
    print("=" * 50)


if __name__ == "__main__":
    main()
