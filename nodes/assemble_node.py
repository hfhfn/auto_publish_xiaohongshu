"""
图片组装节点 — 合并原图与生成图，按场景分组排列（2原图+1生成图）
"""
import os


_MAX_IMAGES = 18  # 小红书单篇笔记最多18张图


def assemble_gallery(state: dict) -> dict:
    """LangGraph节点：合并best_originals和generated_image_paths，按2+1分组排列"""
    best_originals = state.get("best_originals", [])
    generated_paths = state.get("generated_image_paths", [])
    num_prompts = len(state.get("image_prompts", [])) or 3

    print(f"  🖼️ 组装图片画廊: {len(best_originals)} 张原图 + {len(generated_paths)} 张生成图")

    final = []
    seen = set()

    def _add(path):
        """添加图片到final列表，去重"""
        if not path or len(final) >= _MAX_IMAGES:
            return
        norm = os.path.normpath(path)
        if norm not in seen and os.path.isfile(path):
            final.append(path)
            seen.add(norm)

    # 按场景分组排列：每组 2 张原图 + 1 张生成图
    # best_originals: [组0图1, 组0图2, 组1图1, 组1图2, 组2图1, 组2图2]
    for i in range(num_prompts):
        # 该组的2张原图（best_originals中索引 2*i 和 2*i+1）
        orig_idx1 = 2 * i
        orig_idx2 = 2 * i + 1
        if orig_idx1 < len(best_originals):
            _add(best_originals[orig_idx1])
        if orig_idx2 < len(best_originals):
            _add(best_originals[orig_idx2])
        # 该组的1张生成图
        if i < len(generated_paths):
            _add(generated_paths[i])

    # 补充剩余的生成图
    for gen in generated_paths:
        _add(gen)

    # 补充剩余原图
    for orig in best_originals:
        _add(orig)

    print(f"  ✅ 最终画廊: {len(final)} 张图片")
    for idx, p in enumerate(final):
        print(f"    [{idx+1}] {os.path.basename(p)}")

    return {"final_image_paths": final}
