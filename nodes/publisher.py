"""
小红书发布节点 — 使用Playwright操控创作者平台发布笔记
自动启动独立Chrome实例，无需手动操作
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import os
import shutil
import subprocess
import time
import requests
from playwright.async_api import async_playwright
from config import (
    XHS_PUBLISH_URL, CHROME_DEBUG_PORT, CHROME_USER_DATA_DIR,
    ZHIPU_API_KEY, ZHIPU_MODEL, ZHIPU_API_URL,
)


def _find_chrome() -> str:
    """查找系统Chrome路径"""
    candidates = [
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError("未找到Chrome，请安装Chrome或将其加入PATH")


def _is_debug_port_active(port: int) -> bool:
    """检测Chrome调试端口是否已激活"""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ensure_chrome(port: int):
    """确保有一个可用的Chrome调试实例（已有则复用，没有则自动启动）"""
    if _is_debug_port_active(port):
        print(f"  ✅ 检测到已有Chrome调试实例 (端口 {port})")
        return

    print(f"  🚀 自动启动Chrome (独立profile，不影响现有窗口)...")
    chrome_path = _find_chrome()
    CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={CHROME_USER_DATA_DIR}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(15):
        time.sleep(1)
        if _is_debug_port_active(port):
            print(f"  ✅ Chrome已启动 (端口 {port})")
            return
    raise RuntimeError(f"Chrome启动超时，调试端口 {port} 未就绪")


def _llm_decide_collection(topic: str, collection_name: str,
                           existing_names: list[str]) -> dict:
    """调用LLM判断应该加入已有合集还是创建新合集

    返回: {"action": "join", "name": "已有合集名"} 或
          {"action": "create", "name": "新合集名", "desc": "简介"}
    """
    if not existing_names:
        return {"action": "create", "name": collection_name}

    prompt = (
        f'当前要发布一篇关于"{topic}"的小红书旅游笔记。\n'
        f'系统建议的合集名称是："{collection_name}"\n\n'
        f'已有的合集列表：\n'
    )
    for i, name in enumerate(existing_names):
        prompt += f'  {i+1}. {name}\n'
    prompt += (
        '\n重要原则：优先加入已有合集，避免创建重复或相似的合集。\n'
        '判断标准：只要已有合集与本次笔记有任何相关性（同一省份、同一地区、同类主题如都是旅游），就应该加入。\n'
        '只有当所有已有合集都与本次笔记完全无关时（比如一个是美食合集一个是旅游），才创建新合集。\n\n'
        '请用JSON格式回复，不要其他内容：\n'
        '加入已有: {"action":"join","name":"已有合集的完整名称"}\n'
        '创建新的: {"action":"create","name":"新合集名","desc":"50字以内简介"}'
    )

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(ZHIPU_API_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # 提取JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"    ⚠️ LLM合集决策失败: {e}")

    # 回退：默认创建
    return {"action": "create", "name": collection_name}


async def _wait_for_login(page) -> bool:
    """检测是否在登录页，若是则等待用户手动登录完成"""
    if "/login" in page.url:
        print("  ⚠️ 检测到未登录，请在Chrome窗口中完成登录...")
        print("     登录完成后将自动继续。")
        for _ in range(300):
            await page.wait_for_timeout(1000)
            if "/login" not in page.url:
                print("  ✅ 登录成功，继续发布流程")
                await page.wait_for_load_state("networkidle")
                return True
        raise RuntimeError("等待登录超时（5分钟），请重新运行")
    return True


async def _switch_to_image_tab(page):
    """确保切换到"上传图文"标签页（页面默认可能是"上传视频"）"""
    tab_items = page.locator('text=上传图文')
    count = await tab_items.count()
    if count >= 2:
        await tab_items.nth(1).click()
        await page.wait_for_timeout(1000)
    elif count == 1:
        await tab_items.first.click()
        await page.wait_for_timeout(1000)


async def _set_location(page, location: str):
    """设置地点（使用 address-card-select 下拉框 + body 级 popover 选项）"""
    print(f"  📍 设置地点: {location}")
    try:
        # 点击地点区域激活输入
        loc_wrapper = page.locator('.address-card-select')
        await loc_wrapper.click()
        await page.wait_for_timeout(500)

        # 在输入框中填入地点
        loc_input = loc_wrapper.locator('input.d-text')
        await loc_input.fill(location)
        await page.wait_for_timeout(2000)

        # 选项在 body 级 popover 中（不在 select 内部）
        # 找到包含搜索结果的可见 popover
        popovers = page.locator('.d-popover.custom-dropdown-44')
        count = await popovers.count()
        for i in range(count):
            pop = popovers.nth(i)
            if await pop.is_visible():
                first_option = pop.locator('.option-item').first
                if await first_option.count() > 0:
                    await first_option.click(timeout=5000)
                    await page.wait_for_timeout(500)
                    print("  ✅ 地点设置完成")
                    return

        # 备选：直接找所有可见的 .option-item
        visible_option = page.locator('.option-item').first
        await visible_option.click(timeout=5000)
        await page.wait_for_timeout(500)
        print("  ✅ 地点设置完成")
    except Exception as e:
        print(f"  ⚠️ 地点设置失败（可手动设置）: {e}")


async def _handle_collection(page, state: dict):
    """处理合集：LLM判断加入已有合集还是创建新合集"""
    collection_name = state.get("collection_name", "")
    collection_desc = state.get("collection_desc", "")
    topic = state.get("topic", "")

    if not collection_name:
        return

    print(f"  📚 处理合集...")

    try:
        # 检查合集区域是否存在
        wrapper = page.locator('.collection-plugin-wrapper')
        if await wrapper.count() == 0:
            print("    ⚠️ 未找到合集区域")
            return

        # 尝试打开合集下拉框
        # .collection-plugin-choose 是下拉箭头按钮（有已有合集时出现）
        choose_btn = page.locator('.collection-plugin-choose')
        has_choose = await choose_btn.count() > 0 and await choose_btn.is_visible()

        if has_choose:
            await choose_btn.click()
            await page.wait_for_timeout(1000)
        else:
            # 没有下拉按钮，尝试点击整个合集区域
            await wrapper.click()
            await page.wait_for_timeout(1000)

        # 检查弹出层是否出现
        popover = page.locator('.collection-plugin-popover-content')
        has_popover = await popover.count() > 0 and await popover.is_visible()

        if has_popover:
            # 提取合集名列表（去重，保持顺序）
            items = popover.locator('.item .item-label')
            count = await items.count()
            existing_names = []
            seen = set()
            for i in range(count):
                name = await items.nth(i).text_content()
                if name:
                    name = name.strip()
                    if name and name not in seen:
                        existing_names.append(name)
                        seen.add(name)

            print(f"    已有合集: {existing_names}")

            if existing_names:
                # 调用LLM判断
                decision = _llm_decide_collection(topic, collection_name, existing_names)
                print(f"    LLM决策: {decision}")

                if decision["action"] == "join":
                    join_name = decision.get("name", "").strip()
                    if not join_name:
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(500)
                        await _create_collection(page, collection_name, collection_desc)
                    else:
                        joined = await _join_collection_in_popover(
                            page, popover, join_name
                        )
                        if not joined:
                            await page.keyboard.press("Escape")
                            await page.wait_for_timeout(500)
                            await _create_collection(page,
                                                     decision.get("name", collection_name),
                                                     decision.get("desc", collection_desc))
                else:
                    # LLM决定创建新合集 — 点击弹出层底部的"创建合集"
                    footer_create = popover.locator('.popover-footer')
                    if await footer_create.count() > 0:
                        await footer_create.click()
                        await page.wait_for_timeout(1000)
                        await _fill_collection_form(
                            page,
                            decision.get("name", collection_name),
                            decision.get("desc", collection_desc),
                        )
                    else:
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(500)
                        await _create_collection(page,
                                                 decision.get("name", collection_name),
                                                 decision.get("desc", collection_desc))
            else:
                # 弹出层中没有已有合集，点击底部创建
                footer_create = popover.locator('.popover-footer')
                if await footer_create.count() > 0:
                    await footer_create.click()
                    await page.wait_for_timeout(1000)
                    await _fill_collection_form(page, collection_name, collection_desc)
                else:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(500)
                    await _create_collection(page, collection_name, collection_desc)
        else:
            # 弹出层未出现，直接创建
            await _create_collection(page, collection_name, collection_desc)

        await page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ⚠️ 合集处理失败（可手动设置）: {e}")


async def _join_collection_in_popover(page, popover, target_name: str) -> bool:
    """在已打开的弹出层中查找并点击匹配的合集，返回是否成功加入"""
    if not target_name:
        return False

    all_items = popover.locator('.item')
    item_count = await all_items.count()

    # 第一轮：精确匹配或互相包含
    for i in range(item_count):
        item = all_items.nth(i)
        label = item.locator('.item-label')
        if await label.count() > 0:
            text = (await label.text_content() or "").strip()
            if text == target_name or target_name in text or text in target_name:
                try:
                    await item.click()
                except Exception:
                    await label.click()
                await page.wait_for_timeout(1000)
                print(f"  ✅ 已加入合集: {text}")
                return True

    # 第二轮：模糊匹配（关键词交叉）
    print(f"    ⚠️ 未找到精确匹配 '{target_name}'，尝试模糊匹配")
    for i in range(item_count):
        item = all_items.nth(i)
        label = item.locator('.item-label')
        if await label.count() > 0:
            text = (await label.text_content() or "").strip()
            if any(kw in text for kw in target_name.split()):
                try:
                    await item.click()
                except Exception:
                    await label.click()
                await page.wait_for_timeout(1000)
                print(f"  ✅ 已加入合集（模糊匹配）: {text}")
                return True

    return False


async def _create_collection(page, name: str, desc: str):
    """点击创建合集按钮并填写表单"""
    # 先尝试打开合集下拉框，再点击底部"创建合集"
    choose_btn = page.locator('.collection-plugin-choose')
    if await choose_btn.count() > 0 and await choose_btn.is_visible():
        await choose_btn.click()
        await page.wait_for_timeout(1000)
        popover = page.locator('.collection-plugin-popover-content')
        if await popover.count() > 0 and await popover.is_visible():
            footer = popover.locator('.popover-footer')
            if await footer.count() > 0:
                await footer.click()
                await page.wait_for_timeout(1000)
                await _fill_collection_form(page, name, desc)
                return

    # 回退：直接查找"创建合集"文本
    create_btn = page.locator('text=创建合集').first
    if await create_btn.count() > 0:
        await create_btn.click()
        await page.wait_for_timeout(1000)
        await _fill_collection_form(page, name, desc)
    else:
        print(f"    ⚠️ 未找到'创建合集'按钮，跳过合集创建")


async def _fill_collection_form(page, name: str, desc: str):
    """填写创建合集的模态框表单"""
    name_input = page.locator('.d-modal input[placeholder="好的合集名称能吸引更多用户"]')
    try:
        await name_input.wait_for(state="visible", timeout=5000)
    except Exception:
        print("    ⚠️ 创建合集弹窗未出现，跳过")
        return

    await name_input.fill(name[:20])

    if desc:
        desc_input = page.locator('.d-modal textarea[placeholder="简单介绍你的合集"]')
        if await desc_input.count() > 0:
            await desc_input.fill(desc[:50])

    join_btn = page.locator('.d-modal button:has-text("创建并加入")')
    await join_btn.click(timeout=5000)
    await page.wait_for_timeout(2000)
    print(f"  ✅ 已创建并加入合集: {name[:20]}")


async def _publish_async(state: dict):
    """异步执行发布操作"""
    title = state["title"]
    content = state["content"]
    tags = state.get("tags", [])
    location = state.get("location", "")
    image_paths = state.get("final_image_paths", state.get("generated_image_paths", []))
    dry_run = state.get("dry_run", False)

    if not image_paths:
        raise RuntimeError("没有可发布的图片")

    async with async_playwright() as p:
        # 连接已登录的Chrome浏览器
        browser = await p.chromium.connect_over_cdp(
            f"http://127.0.0.1:{CHROME_DEBUG_PORT}"
        )
        context = browser.contexts[0]
        page = await context.new_page()

        # ========== 打开发布页面 ==========
        print("  🌐 打开小红书发布页面...")
        await page.goto(XHS_PUBLISH_URL, wait_until="networkidle", timeout=30000)

        # 检测登录状态，未登录则等待用户手动登录
        await _wait_for_login(page)

        # ========== 切换到图文标签页 ==========
        await _switch_to_image_tab(page)

        # ========== 上传图片 ==========
        print(f"  📸 上传 {len(image_paths)} 张图片...")
        file_input = page.locator('input.upload-input[type="file"]')
        try:
            await file_input.wait_for(state="attached", timeout=10000)
        except Exception:
            file_input = page.locator('input[type="file"]').first
            await file_input.wait_for(state="attached", timeout=10000)

        await file_input.set_input_files(image_paths)

        # 等待图片上传完成
        print("  ⏳ 等待图片处理...")
        await page.wait_for_selector(
            'input[placeholder="填写标题会有更多赞哦"]',
            state="visible", timeout=30000,
        )
        await page.wait_for_timeout(2000)
        print("  ✅ 图片上传完成")

        # ========== 填写标题 ==========
        print(f"  📝 填写标题: {title}")
        title_input = page.locator('input[placeholder="填写标题会有更多赞哦"]')
        await title_input.click()
        await title_input.fill(title)
        await page.wait_for_timeout(500)

        # ========== 填写正文 ==========
        print("  📝 填写正文...")
        content_editor = page.locator('div.tiptap.ProseMirror[contenteditable="true"]')
        try:
            await content_editor.click()
        except Exception:
            content_editor = page.locator('[contenteditable="true"]').first
            await content_editor.click()
        await page.keyboard.type(content, delay=10)
        await page.wait_for_timeout(500)

        # ========== 添加标签 ==========
        if tags:
            print(f"  🏷️ 添加标签: {', '.join(tags[:5])}")
            for tag in tags[:5]:
                await page.keyboard.type(f" #{tag}", delay=10)
                await page.wait_for_timeout(300)
            await page.wait_for_timeout(500)

        # ========== 设置地点 ==========
        if location:
            await _set_location(page, location)

        # ========== 处理合集 ==========
        await _handle_collection(page, state)

        # ========== 发布 ==========
        if dry_run:
            print("  🔍 [DRY RUN] 已填写完成，不执行发布操作")
            print("     请手动检查页面内容，确认无误后手动点击发布")
            # 等待30秒供用户检查，不阻塞进程
            print("     页面将保持30秒后自动关闭...")
            await page.wait_for_timeout(30000)
        else:
            print("  🚀 点击发布...")
            try:
                publish_btn = page.locator('button.custom-button.bg-red')
                try:
                    await publish_btn.click(timeout=5000)
                except Exception:
                    publish_btn = page.locator('button:has-text("发布")').last
                    await publish_btn.click(timeout=5000)
                await page.wait_for_timeout(3000)
                print("  ✅ 发布成功！")
            except Exception as e:
                print(f"  ❌ 发布失败: {e}")
                return {"publish_success": False}

        await page.close()

    return {"publish_success": True}


def publish_to_xiaohongshu(state: dict) -> dict:
    """LangGraph节点：发布笔记到小红书"""
    _ensure_chrome(CHROME_DEBUG_PORT)
    return asyncio.run(_publish_async(state))
