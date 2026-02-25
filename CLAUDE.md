# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated Xiaohongshu (小红书) travel note publisher built on **LangGraph**. The pipeline generates travel copywriting via Zhipu AI, searches real landmark photos via Brave Search, runs image-to-image generation (cloud or local), and publishes to Xiaohongshu creator platform via Playwright browser automation.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install zhipuai torch          # these are commented out in requirements.txt, install separately
playwright install chromium

# Run (dry-run mode - fills form but does not click publish)
python main.py --topic "杭州西湖" --engine jimeng --dry-run

# Run (production - actually publishes)
python main.py --topic "杭州西湖" --engine jimeng

# Engine options: jimeng (cloud, Volcano Engine 即梦4.0) or zimage (local, needs 16GB+ VRAM GPU)
```

**Pre-requisite:** Chrome must be running with `--remote-debugging-port=9222` and the user must be logged into https://creator.xiaohongshu.com before running the publisher.

## Architecture

The project uses a **LangGraph state-machine workflow** defined in `graph.py`:

```
START → generate_content → search_images → [route by engine] → publish → END
                                                ├─ img2img_jimeng
                                                └─ img2img_zimage
```

- **State container:** `PublishState` (TypedDict in `graph.py`) — all nodes read from and write to this shared state dict.
- **Routing:** `route_engine()` in `graph.py` conditionally routes to the selected image generation backend.
- **Node pattern:** Each node in `nodes/` is a function taking `PublishState`, performing its work, and returning a dict of updated state fields.

### Nodes

| Node | File | Purpose |
|------|------|---------|
| `generate_content` | `nodes/content_node.py` | Calls Zhipu GLM-4.5-flash to produce title, content, tags, location, and English image prompts as JSON |
| `search_images` | `nodes/image_search.py` | Queries Brave Image Search API, downloads up to 5 reference photos |
| `img2img_jimeng` | `nodes/image_jimeng.py` | Cloud img2img via Volcano Engine API with V4 HMAC-SHA256 signed requests, async task polling |
| `img2img_zimage` | `nodes/image_zimage.py` | Local img2img via StableDiffusionXLImg2ImgPipeline (diffusers), caches loaded model globally |
| `publish` | `nodes/publisher.py` | Playwright async — connects to Chrome debug port, uploads images, fills form, publishes |

### Config

`config.py` centralizes all settings: output paths, API keys, model names, and the Xiaohongshu creator URL. Volcano Engine credentials are loaded from `AccessKey.txt` (not committed). Zhipu and Brave API keys are currently hardcoded in `config.py`.

### Output

Generated artifacts go to timestamped subdirectories:
- `output/ref/{timestamp}/` — downloaded reference images
- `output/generated/{engine}_{timestamp}/` — AI-generated images

## Key Technical Details

- **Volcano Engine signing** (`image_jimeng.py`): Implements full V4 HMAC-SHA256 request signing (canonical request, string-to-sign, signing key derivation). Changes to this code require careful attention to the signing algorithm.
- **Z-Image pipeline** (`image_zimage.py`): Lazy-loads the diffusers pipeline with a global `_pipe_cache` to avoid reloading the ~7GB model between runs. Images are resized to SDXL-compatible dimensions (multiples of 64).
- **Publisher selectors** (`publisher.py`): Uses multiple CSS selector fallbacks for Xiaohongshu form fields — these are fragile and may break if the platform updates its DOM structure.
- **Content node JSON parsing** (`content_node.py`): The LLM response is expected as JSON; if parsing fails, raw output is written to `debug_raw_output.txt` for diagnosis.

## Language

The project is written for Chinese-language content. Prompts, comments, and generated output are in Chinese. The codebase mixes Chinese comments with English variable names.
