# 自动发布小红书

基于 LangGraph 编排的自动化小红书旅游笔记发布工具。

## 功能流程

1. **目的地调研** — Brave Web Search 搜索旅游攻略，提供真实景点、美食、交通信息
2. **文案生成** — 智谱GLM-4.5-flash生成小红书风格旅游文案（支持DeepSeek降级）
3. **并行处理**（文案精简 与 图片搜索同时进行）
   - **文案精简** — 校验标题/正文字数限制，超限则调用LLM精简
   - **图片搜索** — Brave + Tavily 双引擎搜索真实景点照片，GLM VLM 评估质量
4. **图生图** — 基于真实照片生成写实AI图片
   - 即梦4.0（火山引擎API）
   - Z-Image-Turbo（本地部署，需GPU）
5. **画廊组装** — 合并原图与生成图，按场景分组排列（2原图+1生成图）
6. **自动发布** — Playwright操控小红书创作者平台，DeepSeek辅助合集管理

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
playwright install chromium

# 配置API密钥（复制 .env.example 为 .env 并填写）
cp .env.example .env

# 运行（即梦引擎，试运行模式）
python main.py --topic "杭州西湖" --engine jimeng --dry-run

# 运行（Z-Image本地引擎）
python main.py --topic "杭州西湖" --engine zimage --dry-run

# 正式发布
python main.py --topic "杭州西湖" --engine jimeng

# 从上次失败处恢复
python main.py --resume

# 跳过目的地调研，直接用LLM生成文案
python main.py --topic "杭州西湖" --engine jimeng --skip-research
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--topic` | 旅游主题（完整运行时必填） |
| `--engine` | 图生图引擎：`jimeng`（默认）或 `zimage` |
| `--dry-run` | 试运行模式，填写表单但不点击发布 |
| `--resume` | 从上次失败处恢复执行 |
| `--skip-research` | 跳过目的地调研，直接用LLM生成文案 |

## 项目结构

```
├── main.py                # 入口，支持完整运行和断点恢复
├── graph.py               # LangGraph工作流（并行分支 + assemble_gallery汇合）
├── config.py              # 配置（API密钥通过.env加载）
├── .env.example           # 环境变量模板
├── nodes/
│   ├── research_node.py   # 目的地调研（Brave Web Search）
│   ├── content_node.py    # 文案生成 + 精简（智谱/DeepSeek）
│   ├── image_search.py    # 图片搜索（Brave+Tavily）+ VLM评估
│   ├── image_jimeng.py    # 即梦4.0图生图（火山引擎）
│   ├── image_zimage.py    # Z-Image图生图（本地diffusers）
│   ├── assemble_node.py   # 画廊组装（合并原图+生成图）
│   └── publisher.py       # 小红书发布（Playwright + DeepSeek合集管理）
└── output/                # 输出目录
```
