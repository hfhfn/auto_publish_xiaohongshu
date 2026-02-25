# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated Xiaohongshu (小红书) travel note publisher built on **LangGraph**. The pipeline researches destinations via Brave Web Search, generates travel copywriting via Zhipu AI (with DeepSeek fallback), searches real landmark photos via Brave + Tavily, runs image-to-image generation (cloud or local), and publishes to Xiaohongshu creator platform via Playwright browser automation.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install zhipuai torch          # these are commented out in requirements.txt, install separately
playwright install chromium

# Configure API keys
cp .env.example .env               # then fill in your keys

# Run (dry-run mode - fills form but does not click publish)
python main.py --topic "杭州西湖" --engine jimeng --dry-run

# Run (production - actually publishes)
python main.py --topic "杭州西湖" --engine jimeng

# Resume from last failure
python main.py --resume

# Skip destination research
python main.py --topic "杭州西湖" --engine jimeng --skip-research

# Engine options: jimeng (cloud, Volcano Engine 即梦4.0) or zimage (local, needs 16GB+ VRAM GPU)
```

**Pre-requisite:** Chrome must be running with `--remote-debugging-port=9222` and the user must be logged into https://creator.xiaohongshu.com before running the publisher. The publisher will auto-launch Chrome if no debug instance is detected.

## Architecture

The project uses a **LangGraph state-machine workflow** defined in `graph.py`:

```
START → research_destination → generate_content ─┬→ condense_content → END  (精简文案写入共享state)
                                                  └→ search_images → img2img → assemble_gallery → publish → END
```

- `condense_content` (text branch) and `search_images → img2img` (image branch) run **in parallel** after `generate_content`.
- `condense_content` completes quickly and writes condensed title/content to shared state, then terminates. It does **not** fan-in to `assemble_gallery`.
- `assemble_gallery` has a **single predecessor** (`img2img`), so it runs exactly once with all images available.
- The `img2img` node is a unified dispatcher that internally calls `generate_image_jimeng` or `generate_image_zimage` based on the `engine` state field.

- **State container:** `PublishState` (TypedDict in `graph.py`) — all nodes read from and write to this shared state dict.
- **Node pattern:** Each node in `nodes/` is a function taking `PublishState`, performing its work, and returning a dict of updated state fields.

### Nodes

| Node | File | Purpose |
|------|------|---------|
| `research_destination` | `nodes/research_node.py` | Brave Web Search for travel guides, extracts snippets as research context |
| `generate_content` | `nodes/content_node.py` | Calls Zhipu GLM-4.5-flash (DeepSeek fallback) to produce title, content, tags, location, image prompts, and search queries as JSON |
| `condense_content` | `nodes/content_node.py` | Validates title/content length limits, calls LLM to condense if over limit |
| `search_images` | `nodes/image_search.py` | Brave + Tavily dual-engine image search, downloads candidates, GLM VLM quality scoring |
| `img2img` | `graph.py` (dispatcher) | Unified img2img node — delegates to `generate_image_jimeng` (cloud, Volcano Engine) or `generate_image_zimage` (local, diffusers) based on `engine` state field |
| `assemble_gallery` | `nodes/assemble_node.py` | Merges best originals + generated images in 2+1 grouped order, caps at 18 images |
| `publish` | `nodes/publisher.py` | Playwright async — connects to Chrome debug port, uploads images, fills form, DeepSeek-powered collection management, publishes |

### Config

`config.py` centralizes all settings: output paths, API keys, model names, and the Xiaohongshu creator URL. API keys are loaded from `.env` via `python-dotenv`. Volcano Engine credentials are loaded from `AccessKey.txt` (not committed).

### Output

Generated artifacts go to timestamped subdirectories:
- `output/ref/{timestamp}/` — downloaded reference images
- `output/generated/{engine}_{timestamp}/` — AI-generated images

## Key Technical Details

- **LangChain integration**: Content generation, VLM image evaluation, and DeepSeek collection decisions all use `ChatOpenAI` from `langchain-openai`. Tavily search uses the `tavily-python` SDK. Raw `requests` is only used for Brave Search (no LangChain wrapper for image search) and Volcano Engine (custom signing).
- **Volcano Engine signing** (`image_jimeng.py`): Implements full V4 HMAC-SHA256 request signing (canonical request, string-to-sign, signing key derivation). Changes to this code require careful attention to the signing algorithm.
- **Z-Image pipeline** (`image_zimage.py`): Lazy-loads the diffusers pipeline with a global `_pipe_cache` to avoid reloading the ~7GB model between runs. Images are resized to SDXL-compatible dimensions (multiples of 64).
- **Publisher selectors** (`publisher.py`): Uses multiple CSS selector fallbacks for Xiaohongshu form fields — these are fragile and may break if the platform updates its DOM structure. Includes a defense-in-depth guard against duplicate publishing.
- **Content node JSON parsing** (`content_node.py`): The LLM response is expected as JSON; if parsing fails, raw output is written to `debug_raw_output.txt` for diagnosis.
- **VLM image evaluation** (`image_search.py`): Uses GLM-4V-Flash multimodal model via ChatOpenAI to score images on 7 dimensions with weighted scoring. Includes 429 rate-limit retry logic.

## Language

The project is written for Chinese-language content. Prompts, comments, and generated output are in Chinese. The codebase mixes Chinese comments with English variable names.
