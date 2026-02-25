"""
Z-Image 本地图生图节点 — 使用 Tongyi-MAI/Z-Image-Turbo 进行图生图
需要 GPU（16GB+ VRAM）
"""
import time
import torch
from pathlib import Path
from PIL import Image
from config import ZIMAGE_MODEL_ID, ZIMAGE_STEPS, ZIMAGE_STRENGTH, GEN_IMAGE_DIR


# 全局缓存pipeline，避免重复加载模型
_pipe_cache = {}


def _get_pipeline():
    """懒加载图生图pipeline"""
    if "img2img" not in _pipe_cache:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        print(f"  📦 正在加载Z-Image模型: {ZIMAGE_MODEL_ID} ...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            ZIMAGE_MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # 根据硬件选择设备
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("  🚀 使用 GPU 推理")
        else:
            print("  ⚠️ 未检测到GPU，使用CPU推理（速度较慢）")

        _pipe_cache["img2img"] = pipe

    return _pipe_cache["img2img"]


_CLEANUP_PREFIX = "High quality realistic photograph. Remove all text overlays, watermarks, logos and advertisements. "


def generate_image_zimage(state: dict) -> dict:
    """LangGraph节点：使用Z-Image进行图生图"""
    image_prompts = state.get("image_prompts", [])
    ref_for_gen = state.get("ref_for_gen", [])
    ref_paths = state.get("ref_image_paths", [])

    if not image_prompts:
        image_prompts = [f"A stunning photo of {state['topic']}, realistic photography, 8K"]

    if not ref_for_gen and not ref_paths:
        raise RuntimeError("Z-Image图生图需要参考图，但未找到参考图")

    pipe = _get_pipeline()

    generated_paths = []
    task_dir = GEN_IMAGE_DIR / f"zimage_{int(time.time())}"
    task_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(image_prompts):
        full_prompt = _CLEANUP_PREFIX + prompt
        print(f"  🎨 Z-Image 图生图 [{i+1}/{len(image_prompts)}]: {prompt[:50]}...")

        # 使用专门的img2img参考图（与发布原图不同，保持多样性）
        ref_path = None
        if i < len(ref_for_gen) and ref_for_gen[i]:
            ref_path = ref_for_gen[i]
        elif ref_paths:
            ref_path = ref_paths[i % len(ref_paths)]

        if not ref_path:
            print(f"    ⚠️ 无可用参考图，跳过此prompt")
            continue

        ref_image = Image.open(ref_path).convert("RGB")
        ref_image = _resize_for_sdxl(ref_image)

        # 图生图
        result = pipe(
            prompt=full_prompt,
            image=ref_image,
            strength=ZIMAGE_STRENGTH,
            num_inference_steps=ZIMAGE_STEPS,
            guidance_scale=3.5,
        )
        gen_image = result.images[0]

        path = task_dir / f"gen_{i}.jpg"
        gen_image.save(str(path), quality=95)
        generated_paths.append(str(path))
        print(f"    ✅ 图片已保存: {path.name}")

    return {"generated_image_paths": generated_paths}


def _resize_for_sdxl(image: Image.Image, target_area: int = 1024 * 1024) -> Image.Image:
    """将图片缩放到适合SDXL的尺寸，保持宽高比，并确保尺寸为64的倍数"""
    w, h = image.size
    ratio = (target_area / (w * h)) ** 0.5
    new_w = int(w * ratio) // 64 * 64
    new_h = int(h * ratio) // 64 * 64
    new_w = max(new_w, 512)
    new_h = max(new_h, 512)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
