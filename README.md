# 自动发布小红书 🚀

基于 LangGraph 编排的自动化小红书旅游笔记发布工具。

## 功能流程

1. **文案生成** — 智谱GLM-4.5-flash生成小红书风格旅游文案
2. **图片搜索** — Brave Search API搜索真实景点照片
3. **图生图** — 基于真实照片生成写实AI图片
   - 即梦4.0（火山引擎API）
   - Z-Image-Turbo（本地部署，需GPU）
4. **自动发布** — Playwright操控小红书创作者平台

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
playwright install chromium

# 启动Chrome并登录小红书
chrome --remote-debugging-port=9222
# 手动登录 https://creator.xiaohongshu.com

# 运行（即梦引擎，试运行模式）
python main.py --topic "杭州西湖" --engine jimeng --dry-run

# 运行（Z-Image本地引擎）
python main.py --topic "杭州西湖" --engine zimage --dry-run

# 正式发布
python main.py --topic "杭州西湖" --engine jimeng
```

## 项目结构

```
├── main.py              # 入口
├── graph.py             # LangGraph工作流
├── config.py            # 配置
├── nodes/
│   ├── content_node.py  # 文案生成
│   ├── image_search.py  # 图片搜索
│   ├── image_jimeng.py  # 即梦4.0图生图
│   ├── image_zimage.py  # Z-Image图生图
│   └── publisher.py     # 小红书发布
└── output/              # 输出目录
```
