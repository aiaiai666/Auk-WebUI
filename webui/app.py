# -*- coding: utf-8 -*-
"""
AuK 语音生成与编辑工作台（WebUI 二次开发）
==========================================

基于 `gradio-ui-builder-skill` 的可运行骨架改写的 AuK 推理面板。

与官方 `auk-gradio`（`src/auk/infer/infer_gradio.py`）的区别：

* **懒加载**：启动 WebUI 时不加载任何模型，第一次点「🚀 开始生成」时才构造
  `AukInfer`；加载完成后常驻复用，也可以在「🧰 模型管理」里主动卸载释放显存。
* **产物落盘**：统一写到 `<repo>/outputs/outputs_YYYYMMDD_HHMMSS.wav`，同一秒内
  重复生成自动追加 `_001` / `_002`，绝不重名。
* **默认 CPU offload**：单卡 24 GiB（如 RTX 4090）上 Base 峰值约 24.78 GiB，
  开启 offload 后降到约 16.75 GiB，因此这里默认开启。

运行::

    python -m webui.app                  # 127.0.0.1:7860
    python -m webui.app --host 0.0.0.0   # 允许外部设备访问
    ./start_app.sh 7860                   # 一键启动（自动清端口 + 清显存）

文档对照（skill references）：
    theme-and-color.md       —— CSS 变量与配色（本文件 §CSS）
    layout-and-components.md —— Row/Column/Accordion 与命名（本文件 §LAYOUT）
    events-and-state.md      —— 事件链、gr.Progress、generator（本文件 §EVENTS）
    ui-patterns.md           —— 三段式主区、历史列表、危险确认
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shlex
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from auk.infer.infer_auk import AukInfer, get_gen_duration, save_audio
from auk.infer.infer_gradio import DEMO_EXAMPLE_GROUPS

# 关掉 Gradio 的版本检查上报，避免启动时外连 api.gradio.app
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# ══════════════════════════════════════════════════════════════
# §CONFIG  配置区
# ══════════════════════════════════════════════════════════════

APP_TITLE = "腾讯开源：AuK 语音生成与编辑工作台"
APP_SUBTITLE_L1 = "webUI二次开发 by 科哥 | 微信：312088415 公众号：科哥玩AI"
APP_SUBTITLE_L2 = "承诺永远开源使用 但是需要保留本人版权信息！"
BROWSER_TITLE = f"{APP_TITLE} - 二次开发 by 科哥"

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = Path(os.environ.get("AUK_CKPT_DIR", REPO_ROOT / "ckpts"))
OUTPUT_DIR = Path(os.environ.get("AUK_OUTPUT_DIR", REPO_ROOT / "outputs"))
HISTORY_DIR = OUTPUT_DIR

OUTPUT_PREFIX = "outputs_"  # 产物名 = outputs_YYYYMMDD_HHMMSS[_NNN].wav
# 点示例后把页面滚回顶部（示例在主功能页底部，回填后面板在上方）。
# js 的入参是 inputs + outputs，把 outputs 原样返回，避免控件被瞬间清空。
_SCROLL_TOP_JS = "() => { window.scrollTo({ top: 0, behavior: 'smooth' }); }"

PLACEHOLDER = "── 请选择 ──"

# ══════════════════════════════════════════════════════════════
# §CONFIG  「📖 使用说明」Tab 的正文
# ══════════════════════════════════════════════════════════════

HELP_MARKDOWN = """
## 👋 欢迎使用 AuK 语音工作台

**AuK** 是一个 1.5B 参数的语音生成与编辑基础模型：给它一段音频（或只用文字）+ 一句自然语言指令，
它就能完成文字转语音、内容/声学/副语言编辑、语音增强与说话人分离等任务。
所有任务**共用同一套「指令」接口**，你只需要用大白话描述想做什么。

---

## 🚀 三步上手

**第 1 步 · 选模型**
左上角「模型」下拉选择 `AuK (Base)`（质量优先）或 `AuK-Flash ⚡`（4 步蒸馏，更快）。
首次生成时会自动下载并加载模型，约 1–2 分钟，之后常驻复用。

**第 2 步 · 给输入**
- **参考音频**（可选）：要编辑/克隆的源音频，上传或录音都可以。
- **指令**：用自然语言描述要做什么，例如 `将音调降低 3 个半音`。
- 没有参考音频时只能做「指令 TTS」，这时**必须**在「时长控制」里指定目标时长。

**第 3 步 · 点「🚀 开始生成」**
进度会在状态行实时更新。完成后右侧播放器可直接试听、下载，
文件同时落到项目的 `outputs/` 目录。

> 💡 不想从头写指令？页面底部的**示例区**有 60+ 条现成指令，点一下自动填进面板。

---

## 🧭 页面导览

| Tab | 用来做什么 |
|---|---|
| 🎧 **主功能** | 上传音频 + 写指令 → 生成；**示例列表在本页最底部** |
| 🗂 **历史记录** | 回看/试听 `outputs/` 下所有产物，可一键清空 |
| 🧰 **模型管理** | 查看显存、手动切换模型、卸载释放显存 |
| 📖 **使用说明** | 就是当前这一页 |

---

## ✍️ 指令怎么写

指令越具体，效果越好。按任务类型给几个范例：

| 想做什么 | 指令这样写 |
|---|---|
| 零样本 TTS（克隆音色念新文本） | `用参考音频中相同的音色念出以下内容：「…」` |
| 指令 TTS（只靠文字描述音色） | `请根据下面的音色描述合成语音：「…」，念出：「…」` |
| 改说的内容 | `把「原句」替换成「新句」` |
| 音调 / 语速 / 音量 | `将音调降低 3 个半音` · `将语速调整为 1.5 倍` · `将音量降低 10 分贝` |
| 情感 / 音色 / 口音 | `用开心的语气说这段话` · `保持文字不变，把音色改成：「…」` · `请去掉方言口音` |
| 非语言声音 | `在「…」前加一次呼吸` · `去掉笑声` |
| 增强与分离 | `去掉背景噪声，让人声更干净` · `只保留第一个说话人` |

> ⚠️ **时长优先级**：显式时长 > 参考转写 + 目标文本估算 > 与源音频等长。
> 用 `--gen_seconds` 之外的自动模式时，编辑类任务默认输出与源音频等长。

---

## ❓ 常见问题 FAQ

**Q1：第一次点生成要等很久，是卡了吗？**
不是。首次生成要在显存里加载 Qwen 编码器 + DiT + VAE，约 1–2 分钟。之后同模型复用，几秒就好。

**Q2：切到另一个模型后显存没降，会爆吗？**
不会。WebUI 采用**单模型常驻**：切换时旧模型会被立即卸载。另外注意——
CPU offload 下模型常驻显存只有约 1.4 GiB，真正的大头是**推理峰值**约 10 GiB，
所以这个机制防的是两个峰值叠加爆显存，不是省那 1.4 GiB。

**Q3：AuK-Flash 和 Base 该怎么选？**
Base 质量更高、NFE/CFG 可调；Flash 是 4 步蒸馏模型，固定 NFE=4 且关闭 CFG，适合快速试参数。

**Q4：勾了「使用 Prompt Enhancer」就报错？**
说明还没配 LLM 端点。最省事的办法是用本地已有的 Qwen2.5-Omni-3B：`./start_app.sh` 会在
`.env` 指向 `127.0.0.1` 时自动起 `scripts/local_llm_server.py`，不需要任何云端 key。
也可以展开「Prompt Enhancer 设置」填入 API Key / Base URL / 模型名，或配好 `.env` 后重启。
不配也能用——取消勾选，直接写指令就行。

**Q5：示例的音频格是空的？**
表示该示例的音频文件仓库里没有（例如 Prompt Enhancer 的运行时产物）。
自己上传一段音频再生成即可，示例指令照样有效。

**Q6：提示词在表格里被截断了？**
不会。表格单元格强制完整换行，最多显示 340px 高度，超出部分在单元格内滚动查看。

**Q7：端口 7860 被占用 / 显存被别的程序占了？**
`start_app.sh` 会**无交互**杀掉占用该端口的进程（只杀那一个），轮询确认端口真正释放，
并清理其它占用显存的进程，保证本次运行独占 GPU。

**Q8：生成的文件去哪了？文件名会重复吗？**
统一落在 `outputs/`，命名 `outputs_年月日_时分秒.wav`；同一秒内重复生成会自动追加
`_001`、`_002`，绝不重名。

**Q9：想让手机/别的电脑访问？**
用 `./start_app.sh 7860` 启动（已绑定 `0.0.0.0`），然后访问 `http://<服务器IP>:7860`。

**Q10：生成失败后留下了一个空文件？**
不会。占名的空文件在失败时会被自动清理，历史列表也只显示非空产物。

---

## ⌨️ 快捷键与细节

- **回到页面顶部**：示例在页面底部，点完示例会自动平滑滚回顶部；地址栏加 `/#top` 也可直达。
- **高级参数**（NFE / CFG / 种子）与**时长控制**默认收起在折叠面板里，不影响首屏简洁。
- **音频输入**支持上传与录音两种方式，`type=filepath` 直接把路径交给后端，省内存。
"""

# 单卡 24 GiB 上 Base 峰值约 24.78 GiB（不开）/ 16.75 GiB（开），默认开启。
DEFAULT_CPU_OFFLOAD = True

SAMPLING_NOTE = "Base 可调 NFE / CFG；AuK-Flash 是蒸馏模型，固定 4 步且关闭 CFG。"

# variant 标签 -> 模型产物位置与采样预设（路径在 variant_paths() 里按 CKPT_DIR 拼）
MODEL_VARIANTS: dict[str, dict] = {
    "AuK (Base)": {
        "subdir": "AuK",
        "ckpt_name": "auk_base.safetensors",
        "qwen_subdir": "Qwen2.5-Omni-3B",
        "sampling": {"nfe": 32, "cfg": 2.0, "interactive": True},
    },
    "AuK-Flash ⚡": {
        "subdir": "AuK-Flash",
        "ckpt_name": "auk_flash.safetensors",
        "qwen_subdir": "Qwen2.5-Omni-3B",
        "sampling": {"nfe": 4, "cfg": 0.0, "interactive": False},
    },
}

TASK_HINT = (
    "**支持的任务**：零样本 TTS · 指令 TTS · 内容编辑 · 歌词编辑 · 音调/语速/音量编辑 · "
    "情感/音色/去口音/非语言/耳语编辑 · 语音增强 · 说话人分离 · 人声提取\n\n"
    "**时长优先级**：显式时长 > 参考转写 + 目标文本估算 > 与源音频等长"
)


def variant_paths(label: str) -> dict[str, Path]:
    """把变体标签解析成 ckpt / config / qwen_path 三个绝对路径。"""
    spec = MODEL_VARIANTS[label]
    base = CKPT_DIR / spec["subdir"]
    return {
        "ckpt": base / spec["ckpt_name"],
        "config": base / "config.yaml",  # 发布包自带，且与 ckpt 同目录
        "qwen_path": CKPT_DIR / spec["qwen_subdir"],
    }


def list_variants() -> list[str]:
    """只列出 checkpoint 真实存在的变体（供 UI 启动时探测）。"""
    return [label for label in MODEL_VARIANTS if variant_paths(label)["ckpt"].is_file()]


def list_outputs() -> list[str]:
    """历史清单：只返回文件名，不读文件内容（进页面不加载大音频）。

    过滤掉 0 字节文件：生成被中途取消时会留下一个已占名的空壳。
    """
    if not HISTORY_DIR.is_dir():
        return []
    return sorted(
        (p.name for p in HISTORY_DIR.glob(f"{OUTPUT_PREFIX}*.wav") if p.stat().st_size > 0),
        key=lambda name: (HISTORY_DIR / name).stat().st_mtime,
        reverse=True,
    )


# ══════════════════════════════════════════════════════════════
# §CSS  配色与主题（theme-and-color.md）
# ══════════════════════════════════════════════════════════════

CSS = """
/* ============================================================
   Ocean 主题 · 精致配色覆盖
   中性冷调 slate + 深青强调，专业克制
   ============================================================ */

:root,
.gradio-container {
  --layout-gap: var(--spacing-xl) !important;

  /* —— 中性色阶（冷调 slate）—— */
  --neutral-50:  #f7f9fb;
  --neutral-100: #eef2f6;
  --neutral-200: #e0e6ec;
  --neutral-300: #c7d0da;
  --neutral-400: #97a3b2;
  --neutral-500: #6b7787;
  --neutral-600: #4d5765;
  --neutral-700: #39424f;
  --neutral-800: #232b36;
  --neutral-900: #161c25;
  --neutral-950: #0d1218;

  /* —— 主色：深青 / petrol —— */
  --primary-50:  #edfafa;
  --primary-100: #d3f2f1;
  --primary-200: #a8e4e2;
  --primary-300: #74cecb;
  --primary-400: #45b1af;
  --primary-500: #2b9391;
  --primary-600: #1f7674;
  --primary-700: #1c5f5e;
  --primary-800: #1a4d4c;
  --primary-900: #173f3f;
  --primary-950: #0a2625;

  /* —— 次色：与主色同源的沉稳蓝绿 —— */
  --secondary-50:  #f2f6f8;
  --secondary-100: #e4edf1;
  --secondary-200: #cad9e1;
  --secondary-300: #a3bccb;
  --secondary-400: #6f95aa;
  --secondary-500: #4f7688;
  --secondary-600: #3f5f70;
  --secondary-700: #364e5c;
  --secondary-800: #30424e;
  --secondary-900: #2b3944;
  --secondary-950: #1a242c;
}

/* ===================== 浅色语义 ===================== */
:root,
.gradio-container {
  --body-background-fill: var(--neutral-50);
  --background-fill-primary: #ffffff;
  --background-fill-secondary: var(--neutral-100);

  --body-text-color: var(--neutral-800);
  --body-text-color-subdued: var(--neutral-500);

  --block-background-fill: #ffffff;
  --block-border-color: var(--neutral-200);
  --block-label-text-color: var(--neutral-600);
  --block-title-text-color: var(--neutral-700);

  --border-color-primary: var(--neutral-200);
  --border-color-accent: var(--primary-500);
  --border-color-accent-subdued: var(--primary-200);

  --color-accent: var(--primary-600);
  --color-accent-soft: var(--primary-50);
  --link-text-color: var(--primary-700);
  --link-text-color-hover: var(--primary-800);

  --input-background-fill: #ffffff;
  --input-border-color: var(--neutral-200);
  --input-border-color-focus: var(--primary-400);
  --panel-background-fill: var(--neutral-50);

  --slider-color: var(--primary-600);
  --checkbox-background-color-selected: var(--primary-600);
  --checkbox-border-color-focus: var(--primary-400);

  /* 主按钮：低饱和渐变，克制不浮夸 */
  --button-primary-background-fill: linear-gradient(180deg, var(--primary-500), var(--primary-600));
  --button-primary-background-fill-hover: linear-gradient(180deg, var(--primary-600), var(--primary-700));
  --button-primary-text-color: #ffffff;
  --button-primary-border-color: transparent;

  /* 次按钮：中性描边风格 */
  --button-secondary-background-fill: #ffffff;
  --button-secondary-background-fill-hover: var(--neutral-100);
  --button-secondary-text-color: var(--neutral-700);
  --button-secondary-border-color: var(--neutral-300);

  /* 阴影更柔和 */
  --shadow-drop: 0 1px 2px rgba(16, 24, 32, 0.05);
  --shadow-drop-lg: 0 6px 20px -6px rgba(16, 24, 32, 0.12);
}

/* ===================== 深色语义 ===================== */
.dark {
  --body-background-fill: var(--neutral-950);
  --background-fill-primary: var(--neutral-900);
  --background-fill-secondary: var(--neutral-800);

  --body-text-color: var(--neutral-100);
  --body-text-color-subdued: var(--neutral-400);

  --block-background-fill: var(--neutral-900);
  --block-border-color: var(--neutral-800);
  --block-label-text-color: var(--neutral-300);
  --block-title-text-color: var(--neutral-200);

  --border-color-primary: var(--neutral-800);
  --border-color-accent: var(--primary-500);
  --border-color-accent-subdued: var(--primary-800);

  --color-accent: var(--primary-400);
  --color-accent-soft: rgba(43, 147, 145, 0.14);
  --link-text-color: var(--primary-300);
  --link-text-color-hover: var(--primary-200);

  --input-background-fill: var(--neutral-800);
  --input-border-color: var(--neutral-700);
  --input-border-color-focus: var(--primary-500);
  --panel-background-fill: var(--neutral-900);

  --slider-color: var(--primary-400);
  --checkbox-background-color-selected: var(--primary-500);

  --button-primary-background-fill: linear-gradient(180deg, var(--primary-500), var(--primary-600));
  --button-primary-background-fill-hover: linear-gradient(180deg, var(--primary-400), var(--primary-500));
  --button-primary-text-color: #ffffff;

  --button-secondary-background-fill: var(--neutral-800);
  --button-secondary-background-fill-hover: var(--neutral-700);
  --button-secondary-text-color: var(--neutral-200);
  --button-secondary-border-color: var(--neutral-700);

  --shadow-drop: 0 1px 2px rgba(0, 0, 0, 0.35);
  --shadow-drop-lg: 0 8px 24px -8px rgba(0, 0, 0, 0.5);
}

/* ===================== 轻量收尾 ===================== */
.gradio-container :is(input, textarea, select):focus,
.gradio-container button:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px var(--color-accent-soft);
}

/* 品牌 header：标题与副标题整体居中（需求：web 页面项目标题居中） */
#app-header {
  margin: 2px 0 10px 0; padding: 14px 22px; border-radius: 16px;
  background: linear-gradient(120deg, #2563eb 0%, #4f46e5 45%, #9333ea 100%);
  color: #fff;
  box-shadow: 0 6px 20px rgba(79, 70, 229, .30);
  transition: transform .25s ease, box-shadow .25s ease;
}
#app-header:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(79, 70, 229, .45); }
#app-header .t,
#app-header .s,
#app-header .s2 { text-align: center; }
#app-header .t { font-size: 22px; font-weight: 800; letter-spacing: .5px;
  background: linear-gradient(90deg, #fff, #e0e7ff);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
#app-header .s  { font-size: 13px; font-weight: 600; color: #d7e2ff; }
#app-header .s2 { font-size: 12px; font-weight: 500; color: #eef2ff; opacity: .92; }

/* 组件级微调：只影响挂了对应 class 的组件 */
.compact-audio .audio-container,
.compact-audio .upload-container { min-height: 200px !important; }
#history_list { max-height: 340px; overflow-y: auto; }
/* 历史列表的刷新是瞬时操作，不需要进度条。
   兜底隐藏 status-tracker：否则事件的 tracker 会 absolute 覆盖在选项列表上，
   表现为「历史记录」Tab 一直转圈（主因已由 demo.load 的 show_progress="hidden" 解决）。 */
#history_list [data-testid="status-tracker"] { display: none !important; }
.auk-hint { color: var(--body-text-color-subdued); font-size: 12px; margin-top: -6px; }

/* 示例表格：提示词完整换行显示，绝不省略号截断 */
.auk-examples-table table tbody tr > td:last-child,
.auk-examples-table table tbody tr > td:last-child * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}
.auk-examples-table table tbody tr > td:last-child {
    vertical-align: top;
    max-height: 340px;
    overflow-y: auto !important;
}
"""


# ══════════════════════════════════════════════════════════════
# §BACKEND  业务函数
# ══════════════════════════════════════════════════════════════

_reserved_lock = threading.Lock()


def make_output_path() -> Path:
    """
    预留一个不会重名的输出路径。

    名字固定为 ``outputs_YYYYMMDD_HHMMSS.wav``；同一秒内再次生成时自动追加
    ``_001`` / ``_002``。用 ``touch(exist_ok=False)`` 原子占名，避免 Gradio 队列里
    两个请求抢到同一个文件名。
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with _reserved_lock:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = HISTORY_DIR / f"{OUTPUT_PREFIX}{stamp}.wav"
        serial = 1
        while True:
            if not candidate.exists():
                try:
                    candidate.touch(exist_ok=False)
                    return candidate
                except FileExistsError:
                    pass
            candidate = HISTORY_DIR / f"{OUTPUT_PREFIX}{stamp}_{serial:03d}.wav"
            serial += 1


class EngineManager:
    """
    模型的懒加载、切换与卸载。

    **单模型常驻策略**：RTX 4090 只有 23.5 GiB，单个 Base 变体实测峰值约 10.86 GiB，
    两个变体同时常驻必 OOM。因此加载新变体前会先卸载其它已加载变体，把显存腾出来。
    """

    def __init__(self, single_resident: bool = True) -> None:
        self._engines: dict[str, AukInfer] = {}
        self._lock = threading.Lock()
        self._single_resident = single_resident

    # ------------------------------------------------------------------ 内部
    def _release_locked(self, variant: str) -> bool:
        """卸载一个变体（调用方必须已持有 self._lock）。"""
        engine = self._engines.pop(variant, None)
        if engine is None:
            return False
        # 摘掉 accelerate 的 offload hook，再让引用计数归零后由 gc 回收
        hooks = getattr(engine, "_offload_hooks", None)
        if hooks:
            for hook in hooks:
                for method in ("remove", "detach_hook"):
                    remover = getattr(hook, method, None)
                    if callable(remover):
                        try:
                            remover()
                        except Exception:  # noqa: BLE001 - hook 清理失败不致命
                            pass
        engine.model = None  # type: ignore[assignment]
        engine.vae_model = None  # type: ignore[assignment]
        del engine
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------ 对外
    def get(self, variant: str) -> AukInfer:
        """取（必要时才真正加载）一个变体；单模型策略下会先卸载其它变体。"""
        with self._lock:
            engine = self._engines.get(variant)
            if engine is not None:
                return engine
            if self._single_resident:
                for other in [name for name in self._engines if name != variant]:
                    self._release_locked(other)
            paths = variant_paths(variant)
            engine = AukInfer(
                config_path=str(paths["config"]),
                ckpt_path=str(paths["ckpt"]),
                qwen_path=str(paths["qwen_path"]),
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype="bf16",
                cpu_offload=DEFAULT_CPU_OFFLOAD,
            )
            self._engines[variant] = engine
            return engine

    def activate(self, variant: str) -> str:
        """
        手动切换模型：卸载其它全部，再把选中的变体加载起来（常驻显存）。

        供模型管理页「🔀 切换到此模型」按钮调用。
        """
        with self._lock:
            released = [name for name in self._engines if name != variant]
            for other in released:
                self._release_locked(other)
        if released:
            return f"🧹 已卸载 {'、'.join(released)}，显存已释放。"
        return "当前没有其它已加载的模型，无需释放。"

    def release_others(self, variant: str) -> str:
        """
        只卸载除 variant 以外的变体，**不加载 variant**。

        主功能页切换下拉框时调用：既立刻释放旧模型显存，又保持懒加载
        （新模型仍等到第一次生成时才真正载入）。
        """
        with self._lock:
            released = [name for name in self._engines if name != variant]
            for other in released:
                self._release_locked(other)
        if released:
            return f"🧹 已卸载 {'、'.join(released)}，显存已释放；**{variant}** 将在首次生成时载入。"
        return "（没有其它已加载的模型）"

    def unload(self, variant: str) -> bool:
        with self._lock:
            return self._release_locked(variant)

    def unload_all(self) -> int:
        with self._lock:
            return sum(1 for variant in list(self._engines) if self._release_locked(variant))

    def loaded(self) -> list[str]:
        return sorted(self._engines)

    def status_markdown(self) -> str:
        available = list_variants()
        lines: list[str] = []
        if not available:
            return (
                f"❌ 未找到任何 checkpoint（`{CKPT_DIR}`）。请先运行：\n\n"
                "```bash\nexport HF_ENDPOINT=https://hf-mirror.com\n"
                "python scripts/download_models.py\n```"
            )
        loaded = self.loaded()
        lines.append(f"**模型目录**：`{CKPT_DIR}`")
        lines.append("**可用变体**：" + "、".join(f"{name}{' ✅已加载' if name in loaded else ''}" for name in available))
        lines.append(f"**显存占用**：{vram_summary()}")
        if len(available) > 1:
            lines.append("\n> ⚠️ 单卡 23.5 GiB 容不下两个变体同时常驻，切换模型时会**自动卸载旧模型**以释放显存。")
        return "\n\n".join(lines)


ENGINES = EngineManager()


def vram_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA 不可用（CPU 模式）"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return f"{used / 1024**3:.2f} / {total / 1024**3:.2f} GiB"


def update_sampling_controls(variant: str):
    preset = MODEL_VARIANTS.get(variant, {}).get("sampling", {"nfe": 32, "cfg": 2.0, "interactive": True})
    interactive = preset["interactive"]
    return (
        gr.update(value=preset["nfe"], interactive=interactive),
        gr.update(value=preset["cfg"], interactive=interactive),
    )


def run_task(
    variant: str,
    audio: str | None,
    instruction: str,
    gen_seconds: float,
    ref_text: str,
    gen_text: str,
    nfe: int,
    cfg: float,
    seed: int | None,
    use_pe: bool,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
    progress=gr.Progress(),
):
    """
    主任务：generator 函数，每 yield 一次 = 往前端推一次中间态。

    要点（skill · events-and-state.md §四）：
      * yield 的元素个数必须和 outputs 一致，不变的那个用 gr.update()；
      * 前置校验放在最前，失败即 yield + return。
    """
    # ---- 前置校验 ----
    if not variant or variant == PLACEHOLDER:
        yield gr.update(value="🙌 请先选择模型变体。"), gr.update(), gr.update()
        return
    if not instruction or not instruction.strip():
        yield gr.update(value="🙌 请先填写指令（Instruction）。"), gr.update(), gr.update()
        return
    if not list_variants():
        yield (
            gr.update(value="❌ 未找到 checkpoint，请先运行 `python scripts/download_models.py`。"),
            gr.update(),
            gr.update(),
        )
        return
    if not audio and not gen_seconds and not gen_text:
        yield (
            gr.update(value="🙌 无参考音频时（指令 TTS）必须指定目标时长或目标文本。"),
            gr.update(),
            gr.update(),
        )
        return

    instruction = instruction.strip()
    out_path = make_output_path()  # 先占名，失败时再删，避免留下半截文件
    pe_note = ""
    prepared = None  # PE 的临时音频所有权在本函数，生成结束后才清理

    try:
        # ---- 懒加载模型：第一次生成时才真正吃显存 ----
        yield (
            gr.update(value=f"⏳ 正在加载 **{variant}**（首次约 1–2 分钟，之后常驻复用）…"),
            gr.update(),
            gr.update(),
        )
        engine = ENGINES.get(variant)

        target_seconds = float(gen_seconds or 0)

        if use_pe:
            yield (
                gr.update(value="⏳ Prompt Enhancer 正在分析任务与准备音频…"),
                gr.update(),
                gr.update(),
            )
            instruction, audio, ref_text, gen_text, target_seconds, pe_note, prepared = _prepare_with_pe(
                instruction,
                audio,
                ref_text,
                gen_text,
                target_seconds,
                llm_api_key,
                llm_base_url,
                llm_model,
            )

        resolved_seconds = get_gen_duration(
            audio=(audio or None),
            ref_text=(ref_text or None),
            gen_text=(gen_text or None),
            gen_seconds=(target_seconds if target_seconds > 0 else None),
        )

        yield (
            gr.update(value=f"⏳ 推理中（目标 {resolved_seconds or '与源等长'} 秒）…"),
            gr.update(),
            gr.update(),
        )
        progress(0.45, desc="推理中")

        content = [{"type": "text", "text": instruction}]
        if audio:
            content.append({"type": "audio", "audio": audio})
        messages = [{"role": "user", "content": content}]

        out_audio, sample_rate = engine.generate(
            messages,
            audio=(audio or None),
            gen_seconds=resolved_seconds,
            nfe=int(nfe),
            cfg_strength=float(cfg),
            seed=(int(seed) if seed is not None else None),
        )

        progress(0.9, desc="写出音频")
        save_audio(out_audio, sample_rate, str(out_path))
    except gr.Error:
        # gr.Error 由 Gradio 原样呈现为错误弹窗，不要吞掉
        out_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - 统一回报给前端，并清掉占名的空文件
        out_path.unlink(missing_ok=True)
        yield (
            gr.update(value=f"❌ 生成失败：{type(exc).__name__}: {exc}"),
            gr.update(),
            gr.update(),
        )
        return
    finally:
        # PE 可能在 /tmp 留下裁剪/归一化后的音频，必须等 engine.generate 读完之后再删。
        if prepared is not None:
            prepared.cleanup()

    seconds = out_audio.shape[-1] / sample_rate
    info = f"**已保存**：`{out_path}`\n\n时长 {seconds:.2f}s · 采样率 {sample_rate} Hz · 变体 {variant}" + (
        f"\n\n{pe_note}" if pe_note else ""
    )
    yield (
        gr.update(value=f"✅ 完成 · {seconds:.1f}s · 已写入 `{out_path.name}`"),
        gr.update(value=str(out_path)),
        gr.update(value=info),
    )


def _prepare_with_pe(
    instruction: str,
    audio: str | None,
    ref_text: str,
    gen_text: str,
    target_seconds: float,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
) -> tuple[str, str | None, str, str, float, str, object]:
    """
    调用 Prompt Enhancer 预处理指令。

    Prompt Enhancer 的核心是**用 LLM 做任务分类与参数抽取**，因此必须有一个
    OpenAI 兼容端点；没有任何本地兜底路径。缺配置时给出可操作的提示，
    而不是把底层 ValueError 直接抛到用户脸上。

    第七个返回值是 ``PromptEnhancerOutput``，**所有权交给调用方**：PE 某些任务
    （VAD 裁剪、响度归一化）会写临时 wav 并把它作为 ``audio`` 返回，若在此处
    ``cleanup()``，调用方拿到的就是一个已被删除的路径。所以清理必须延后到
    真正消费完 ``audio`` 之后——与 ``infer_gradio.run_generate`` 的做法一致。
    """
    from auk.infer.pe import PromptEnhancer, PromptEnhancerError

    # 页面三个输入框留空时回落到 .env 的 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME，
    # 与 llm_ready() 同一套判定；只查入参会让「用环境变量」这条路径永远不可达。
    if not llm_ready(llm_api_key, llm_base_url, llm_model):
        raise gr.Error(
            "**未配置 Prompt Enhancer 所需的 LLM。**\n\n"
            "PE 靠 LLM 来识别任务、抽取参数。配一个 OpenAI 兼容端点即可，三种方式：\n\n"
            "方式一 · 本地零成本（无需任何云端 key）：\n"
            "```bash\n./start_app.sh 7860\n```\n"
            "`.env` 里的 `LLM_BASE_URL` 指向 `http://127.0.0.1:8000/v1` 时，"
            "启动脚本会自动拉起 `scripts/local_llm_server.py`，"
            "复用已下载的 `ckpts/Qwen2.5-Omni-3B`。\n\n"
            "方式二 · 云端 / 自建端点：\n"
            "```bash\ncp .env.example .env      # 填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME\n"
            "set -a; source ./.env; set +a\n./start_app.sh 7860\n```\n\n"
            "方式三 · 在下方「Prompt Enhancer 设置」里直接填（即刻生效，无需重启）\n\n"
            "`LLM_BASE_URL` 填 API 根地址，不要带 `/chat/completions`。\n\n"
            "> 如果只是想生成语音，把「使用 Prompt Enhancer」取消勾选即可——"
            "所有任务都可以直接写指令完成，PE 只是可选的增强步骤。"
        )

    prepared = None
    handed_off = False
    try:
        prepared = PromptEnhancer(llm_api_key=llm_api_key, llm_base_url=llm_base_url, llm_model=llm_model).prepare(
            instruction,
            audio,
            target_duration=target_seconds if target_seconds > 0 else None,
        )
        task = str(getattr(prepared, "task_type", "") or "")
        subtype = getattr(prepared, "operation_subtype", None)
        if subtype:
            task = f"{task} / {subtype}"
        asr_text = getattr(getattr(prepared, "asr", None), "text", None) or ""
        note = f"**PE 识别任务**：{task or '未知'}"
        if asr_text:
            note += f"\n\n**ASR 转写**：{asr_text}"
        result = (
            prepared.instruction,
            prepared.audio,
            prepared.ref_text,
            prepared.gen_text,
            (target_seconds if target_seconds > 0 else prepared.gen_seconds),
            note,
            prepared,
        )
        handed_off = True
        return result
    except PromptEnhancerError as exc:
        raise gr.Error(f"Prompt Enhancer 失败：{exc}") from None
    except FileNotFoundError as exc:
        raise gr.Error(f"Prompt Enhancer 找不到音频文件：{exc}") from None
    except Exception as exc:  # noqa: BLE001 - LLM 网络/鉴权错误统一呈现
        raise gr.Error(
            f"Prompt Enhancer 调用 LLM 失败：{type(exc).__name__}: {exc}\n\n"
            "请检查 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME` 是否正确、端点是否可达。"
        ) from None
    finally:
        # 只有失败路径（没交出去）才在这里清理；成功时由 run_task 负责。
        if prepared is not None and not handed_off:
            prepared.cleanup()


def refresh_history():
    """历史 Tab：只给 Radio 喂文件名，不加载音频。"""
    choices = list_outputs()
    if not choices:
        return gr.update(choices=[], value=None), "📭 还没有历史产物，先去「🎧 主功能」生成一个。"
    return gr.update(choices=choices, value=choices[0]), f"共 {len(choices)} 个产物，**选中即播放**。"


def pick_history(name: str):
    """选中某条历史才真正读盘。"""
    if not name:
        return gr.update(), gr.update()
    return gr.update(value=str(HISTORY_DIR / name)), gr.update(value=f"正在播放：`{name}`")


def delete_history(confirmed: bool):
    """危险操作：没勾确认框时给出可见提示，不要静默失败。"""
    if not confirmed:
        return gr.update(), gr.update(), "⚠️ 请先勾选确认框再删除。"
    removed = 0
    if HISTORY_DIR.is_dir():
        for path in HISTORY_DIR.glob(f"{OUTPUT_PREFIX}*.wav"):
            path.unlink(missing_ok=True)
            removed += 1
    return gr.update(choices=[], value=None), gr.update(value=None), f"🗑 已删除 {removed} 个产物"


def refresh_engines():
    """模型管理 Tab：状态 + 卸载按钮可用性。"""
    return ENGINES.status_markdown(), gr.update(interactive=bool(ENGINES.loaded()))


def unload_all_engines():
    count = ENGINES.unload_all()
    message = f"🧹 已卸载 {count} 个模型，显存已释放。" if count else "当前没有已加载的模型。"
    return ENGINES.status_markdown(), gr.update(interactive=False), message


def switch_engine(variant: str):
    """
    模型管理页「切换到此模型」：卸载其它变体释放显存，再把选中的变体加载起来。
    """
    if not variant or variant == PLACEHOLDER:
        return ENGINES.status_markdown(), "🙌 请先选择要切换的变体。"
    if variant not in list_variants():
        return ENGINES.status_markdown(), f"❌ 变体 {variant} 的权重不存在，请先下载。"

    released = ENGINES.activate(variant)
    if variant not in ENGINES.loaded():
        try:
            ENGINES.get(variant)  # 真正加载（会先把其它变体卸干净）
        except Exception as exc:  # noqa: BLE001 - 回报给前端，不中断 UI
            return ENGINES.status_markdown(), f"❌ 加载 {variant} 失败：{type(exc).__name__}: {exc}"
    message = f"{released}\n\n✅ 已切换到 **{variant}**（已常驻显存，可直接生成）。"
    return ENGINES.status_markdown(), message


def llm_ready(key: str = "", base_url: str = "", model: str = "") -> bool:
    """UI 填的值或环境变量三者齐全，PE 才可用。"""

    def from_env(env: str) -> bool:
        return bool(str(key or os.environ.get(env) or "").strip())

    def from_base(env: str) -> bool:
        return bool(str(base_url or os.environ.get(env) or "").strip())

    def from_model(env: str) -> bool:
        return bool(str(model or os.environ.get(env) or "").strip())

    return from_env("LLM_API_KEY") and from_base("LLM_BASE_URL") and from_model("LLM_MODEL_NAME")


def check_pe(use_pe: bool, key: str = "", base_url: str = "", model: str = ""):
    """
    勾选/取消 PE 时的即时反馈。

    没有 LLM 配置就当场说清楚缺什么、怎么补，而不是等用户上传完音频、
    点了生成才在 30 秒后看到一个底层 ValueError。
    """
    if not use_pe:
        return gr.update(value=""), gr.update(info="自动识别任务并转写音频，需要 LLM / ASR 凭据。")
    if llm_ready(key, base_url, model):
        return gr.update(value="✅ Prompt Enhancer 已启用，LLM 配置就绪。"), gr.update(info="自动识别任务并转写音频。")
    return (
        gr.update(
            value=(
                "⚠️ **Prompt Enhancer 还不可用**：未检测到 LLM 配置"
                "（`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME`）。\n\n"
                "PE 靠 LLM 识别任务、抽取参数。三种配法：\n\n"
                "1. **本地零成本**：`./start_app.sh` 会在 `.env` 指向 `127.0.0.1` 时自动拉起 "
                "`scripts/local_llm_server.py`（复用已下载的 `ckpts/Qwen2.5-Omni-3B`），无需云端 key；\n"
                "2. **在本页填**：展开下方「Prompt Enhancer 设置」直接填，即刻生效；\n"
                "3. **用环境变量**：`cp .env.example .env` 填好后重启 `./start_app.sh`。\n\n"
                "> 不配也能用：取消勾选，所有任务都可以直接写指令完成。"
            )
        ),
        gr.update(info="⚠️ 缺少 LLM 配置，点展开「Prompt Enhancer 设置」补齐。"),
    )


def release_other_engines(variant: str):
    """
    主功能页切换模型下拉：卸掉其它变体释放显存，但不预加载新变体（保持懒加载）。
    """
    if not variant or variant == PLACEHOLDER:
        return ENGINES.status_markdown(), "⏸ 等待生成"
    return ENGINES.status_markdown(), ENGINES.release_others(variant)


def unload_one_engine(variant: str):
    if not variant or variant == PLACEHOLDER:
        return ENGINES.status_markdown(), gr.update(interactive=bool(ENGINES.loaded())), "🙌 请先选择要卸载的变体。"
    if ENGINES.unload(variant):
        message = f"🧹 已卸载 **{variant}**，显存已释放。"
    else:
        message = f"**{variant}** 当前并未加载。"
    return ENGINES.status_markdown(), gr.update(interactive=bool(ENGINES.loaded())), message


def load_demo_example(index: int | None, rows: list[dict]):
    """
    把选中的示例填进主功能面板。

    PE 勾选框**永远不自动勾上**：即便示例本身是 PE 命令，也留给你手动决定，
    避免没配 LLM 时一点示例就直接报错。
    音频为 None 时该槽位留空，由用户自己上传。
    """
    if index is None or not 0 <= int(index) < len(rows):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    row = rows[int(index)]
    return (
        row.get("audio") or None,
        row.get("instruction") or "",
        float(row.get("gen_seconds") or 0),
        False,
    )


def _audio_or_blank(relative: str | None) -> str | None:
    """仓库相对路径 -> 绝对路径；文件缺失返回 None（示例区留空，让用户补音频）。"""
    if not relative:
        return None
    candidate = REPO_ROOT / relative
    return str(candidate) if candidate.is_file() else None


def builtin_example_rows() -> dict[str, dict[str, list[dict]]]:
    """把内置示例表转成统一结构（音频缺失的条目照样保留，音频位留空，提示词中文化）。"""
    result: dict[str, dict[str, list[dict]]] = {}
    for category, tasks in DEMO_EXAMPLE_GROUPS.items():
        for task, items in tasks.items():
            result.setdefault(zh_label(category), {})[zh_label(task)] = [
                {
                    "audio": _audio_or_blank(audio),
                    "instruction": zh_instruction(instruction),
                    "gen_seconds": duration,
                }
                for audio, instruction, duration in items
            ]
    return result


def readme_example_rows() -> dict[str, list[dict]]:
    """README.md 示例 -> 统一结构，同样保留音频缺失的条目，提示词中文化。"""
    result: dict[str, list[dict]] = {}
    for heading, items in parse_readme_examples().items():
        result[zh_label(heading)] = [
            {
                "audio": _audio_or_blank(item.get("audio")),
                "instruction": zh_instruction(item.get("instruction") or ""),
                "gen_seconds": item.get("gen_seconds"),
            }
            for item in items
        ]
    return result


def _example_samples(rows: list[dict], only_instruction: bool = False) -> list[list]:
    """Dataset 的样例数据；音频缺失时该列给 None，前端显示为空上传框。"""
    if only_instruction:
        return [[row["instruction"]] for row in rows]
    return [[row.get("audio"), row["instruction"]] for row in rows]


# ══════════════════════════════════════════════════════════════
# §BACKEND  中文化（示例 Tab 的标题与提示词）
# ══════════════════════════════════════════════════════════════

# 分类 / 任务名 -> 中文标签
_ZH_LABELS: dict[str, str] = {
    # README 小节标题
    "Prompt Enhancer": "提示词增强器",
    "Command-line inference": "命令行推理",
    "CLI Examples": "命令行示例",
    "Content editing": "内容编辑",
    "Speech enhancement / separation": "语音增强 / 分离",
    "Zero-shot TTS": "零样本 TTS",
    "Lower VRAM usage (CUDA only)": "降低显存占用（仅 CUDA）",
    "Interactive Gradio demo": "Gradio 交互演示",
    "ComfyUI": "ComfyUI",
    "Python API": "Python API",
    "Quick Start": "快速开始",
    "Installation": "安装",
    "Download the weights": "下载权重",
    "Fine-tuning": "微调",
    # 内置分类
    "Speech Generation": "语音生成",
    "Content Editing": "内容编辑",
    "Acoustic Editing": "声学编辑",
    "Paralinguistic Editing": "副语言编辑",
    "Enhancement & Separation": "增强与分离",
    # 内置任务
    "Instruct TTS": "指令 TTS",
    "Speech Content Editing": "语音内容编辑",
    "Lyric Editing": "歌词编辑",
    "Pitch Editing": "音调编辑",
    "Speed Editing": "语速编辑",
    "Volume Editing": "音量编辑",
    "Emotion Editing": "情感编辑",
    "Timbre Editing": "音色编辑",
    "De-accent": "去口音",
    "Nonverbal Editing": "非语言编辑",
    "Whisper Conversion": "耳语转换",
    "Speech Enhancement": "语音增强",
    "Speech Separation": "说话人分离",
    "Vocal Extraction": "人声提取",
    "Audio Quality Enhancement": "音质增强",
}


def _strip_outer_quotes(text: str) -> str:
    """只剥掉最外层的一对引号——避免把 it's 里的撇号当成引号。"""
    stripped = text.strip()
    for open_q, close_q in (("“", "”"), ("'", "'"), ('"', '"'), ("「", "」")):
        if len(stripped) >= 2 and stripped.startswith(open_q) and stripped.endswith(close_q):
            return stripped[1:-1].strip()
    return stripped


def _norm_curly(text: str) -> str:
    """把中文弯引号 “ ” 统一成 「 」；单引号不动（撇号有歧义）。"""
    return re.sub(r"“([^“”]+)”", r"「\1」", text)


# 指令里的英文"框架"改写成中文；引号内的台词 / 音色描述等 payload 原样保留
_ZH_INSTRUCTION_RULES: list[tuple[re.Pattern, object]] = [
    (
        re.compile(r"^Say the following in the voice described here:\s*(.+?),\s*and say:\s*(.+)$", re.S),
        lambda m: "请根据下面的音色描述合成语音：「%s」，念出：「%s」"
        % (_strip_outer_quotes(m.group(1)), _strip_outer_quotes(m.group(2))),
    ),
    (
        re.compile(r"^Say the following with the same voice:\s*(.+)$", re.S),
        lambda m: "用参考音频中相同的音色念出以下内容：「%s」" % _strip_outer_quotes(m.group(1)),
    ),
    (
        re.compile(r"^Keep the words and change the timbre to:\s*(.+)$", re.S),
        lambda m: "保持文字不变，把音色改成：「%s」" % _strip_outer_quotes(m.group(1)),
    ),
]

# 没有 payload 需要保留的指令，整条替换
_ZH_INSTRUCTIONS: dict[str, str] = {
    "Convert this whisper into normal speech while preserving the speaker and content.": "把这段耳语转换成正常说话的声音，保持说话人和内容不变。",
    "Convert this whispered speech into normal speech.": "把这段耳语音频转换成正常说话的声音。",
    "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.": "把「but accepting what we cannot have」替换成「and living well with dreams unmet」。",
    "Replace “rear view” with “like you” in the lyrics": "把歌词里的「rear view」替换成「like you」。",
    "Say this in a happy tone": "用开心的语气说这段话。",
    "Say this in a sad tone": "用悲伤的语气说这段话。",
    "Say this in a angry tone": "用愤怒的语气说这段话。",
    "Say this in a afraid tone": "用害怕的语气说这段话。",
    "Add a breath before “We tested”": "在「We tested」前加一次呼吸。",
    "Add a sneeze before “only one of them”": "在「only one of them」前加一个喷嚏。",
    "Add a pause after “only one of them”": "在「only one of them」后加一个停顿。",
    "Remove the humming": "去掉哼唱声。",
    "Remove the hiss": "去掉嘶嘶声。",
    "Remove the sobbing": "去掉啜泣声。",
    "Add a sigh before “这个月”": "在「这个月」前加一声叹气。",
    "Add a filler before “主要的缺口”": "在「主要的缺口」前加一个口头禅。",
    "Add a cough before “再决定”": "在「再决定」前加一声咳嗽。",
    "Remove the laughter": "去掉笑声。",
    "Remove the gasp of surprise": "去掉惊讶的抽气声。",
    "Remove the sharp inhale": "去掉急促的吸气声。",
    "Turn this into a whisper": "把这段音频转换成耳语。",
    "Remove the background noise and make the voice cleaner": "去掉背景噪声，让人声更干净。",
    "Keep only the speaker who says “get what”": "只保留说「get what」的那个说话人。",
    "Keep only the first speaker": "只保留第一个说话人。",
    "Keep only the speaker who says “警队规矩”": "只保留说「警队规矩」的那个说话人。",
    "Keep only the second speaker": "只保留第二个说话人。",
    "Extract the vocals and remove the accompaniment": "提取人声，去掉伴奏。",
    "Improve the audio quality and make it clearer": "提升音质，让声音更清晰。",
}


def zh_label(name: str) -> str:
    """分类 / 任务名中文化，未收录的原样返回。"""
    return _ZH_LABELS.get(name.strip(), name.strip())


def zh_instruction(text: str) -> str:
    """指令中文化：英文框架换成中文，引号内的台词 / 描述等 payload 原样保留。"""
    if not text:
        return text
    if text in _ZH_INSTRUCTIONS:
        return _ZH_INSTRUCTIONS[text]
    for pattern, make in _ZH_INSTRUCTION_RULES:
        match = pattern.match(text)
        if match:
            return _norm_curly(make(match))  # type: ignore[operator]
    # 本来就含中文的（音调/语速/音量/去口音等）只统一引号风格
    return _norm_curly(text) if re.search(r"[\u4e00-\u9fff]", text) else text


# ══════════════════════════════════════════════════════════════
# §BACKEND  README 示例解析
# ══════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*?)\s*$")
_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_CMD_RE = re.compile(r"^python\d?\s+\S*pe\.py\b")
_VALUE_ARGS = ("--audio", "--instruction", "--gen_seconds", "--gen_text", "--ref_text")


def _parse_readme_command(block: str) -> dict | None:
    """从一段 bash 代码块里解析 auk-infer / pe.py 调用，不相关则返回 None。"""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("auk-infer") or _CMD_RE.match(stripped):
            start = index
            break
    if start is None:
        return None

    body = "\n".join(lines[start:]).replace("\\\n", " ")
    try:
        tokens = shlex.split(body, comments=False, posix=True)
    except ValueError:
        return None

    parsed: dict = {"audio": None, "instruction": None, "gen_seconds": None, "gen_text": None, "ref_text": None}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _VALUE_ARGS and index + 1 < len(tokens):
            parsed[token.lstrip("-")] = tokens[index + 1]
            index += 2
            continue
        index += 1
    if not parsed["instruction"]:
        return None

    try:
        parsed["gen_seconds"] = float(parsed["gen_seconds"]) if parsed["gen_seconds"] else None
    except (TypeError, ValueError):
        parsed["gen_seconds"] = None
    return parsed


def parse_readme_examples(readme_path: Path | None = None) -> dict[str, list[dict]]:
    """
    解析 README.md 里的全部示例命令，按小节标题归类。

    这样「示例」Tab 永远和文档保持一致，文档加了新示例，刷新页面就有。
    """
    path = readme_path or (REPO_ROOT / "README.md")
    if not path.is_file():
        return {}

    groups: dict[str, list[dict]] = {}
    current = "未分类"
    in_fence = False
    lang = ""
    buffer: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            if not in_fence:
                in_fence, lang, buffer = True, fence.group(1), []
            else:
                in_fence = False
                if lang in ("bash", "sh", "shell"):
                    parsed = _parse_readme_command("\n".join(buffer))
                    if parsed:
                        groups.setdefault(current, []).append(parsed)
                buffer = []
            continue
        if in_fence:
            buffer.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            current = heading.group(2).strip()
            continue
        bold = _BOLD_RE.match(line)
        if bold:
            current = bold.group(1).strip()

    return groups


def demo_dataset_samples(task: str, examples: list[tuple[str | None, str, float]]) -> list[list]:
    if task == "Instruct TTS":
        return [[instruction] for _, instruction, _ in examples]
    return [
        [audio, instruction] for audio, instruction, _ in examples
    ]  # ══════════════════════════════════════════════════════════════


# §LAYOUT  界面装配
# ══════════════════════════════════════════════════════════════


def build_demo() -> gr.Blocks:
    available_variants = list_variants()
    initial_variant = available_variants[0] if available_variants else PLACEHOLDER
    initial_sampling = MODEL_VARIANTS.get(initial_variant, {}).get("sampling", {"nfe": 32, "cfg": 2.0, "interactive": True})

    with gr.Blocks(title=BROWSER_TITLE) as demo:
        # ---- 品牌 header：标题居中 + 两行副标题 ----
        gr.HTML(
            f'<div id="app-header">'
            f'<div class="t">{APP_TITLE}</div>'
            f'<div class="s">{APP_SUBTITLE_L1}</div>'
            f'<div class="s s2">{APP_SUBTITLE_L2}</div>'
            f"</div>",
            elem_id="top",
        )

        if not available_variants:
            gr.Markdown(
                f"⚠️ **未检测到模型权重**（`{CKPT_DIR}`）。请先运行：\n\n"
                "```bash\nexport HF_ENDPOINT=https://hf-mirror.com\n"
                "python scripts/download_models.py\n```"
            )

        with gr.Tabs():
            # ======================= 主功能 ======================= #
            with gr.Tab("🎧 主功能", id="tab-main"):
                gr.Markdown(TASK_HINT)
                with gr.Row(equal_height=True):
                    # 左侧窄列：资源选择（模型 / 参考音频），scale=1
                    with gr.Column(scale=1):
                        with gr.Row():
                            variant_dd = gr.Dropdown(
                                choices=available_variants or [PLACEHOLDER],
                                value=initial_variant,
                                show_label=False,
                                container=False,
                                scale=1,
                            )
                        in_audio = gr.Audio(
                            label="参考音频（可选；留空 = 指令 TTS）",
                            sources=["upload", "microphone"],
                            type="filepath",
                            elem_classes=["compact-audio"],
                        )
                        use_pe = gr.Checkbox(
                            value=False,
                            label="使用 Prompt Enhancer",
                            info="默认关闭。开启后需配置 LLM（见下方「Prompt Enhancer 设置」）才能用。",
                        )
                        pe_hint = gr.Markdown("")
                        gen_btn = gr.Button("🚀 开始生成", variant="primary")

                        with gr.Accordion("Prompt Enhancer 设置", open=False):
                            gr.Markdown(
                                "PE 通过 LLM 识别任务并抽取参数，需一个 OpenAI 兼容端点。\n\n"
                                "留空则读环境变量 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME`"
                                "（即 `.env` 里那份）。在此填写只保存在当前浏览器会话，不会写盘。\n\n"
                                "> 没有云端 key？把 `.env` 的 `LLM_BASE_URL` 指向 `http://127.0.0.1:8000/v1`，"
                                "`./start_app.sh` 会自动起本地服务复用 `ckpts/Qwen2.5-Omni-3B`。"
                            )
                            pe_llm_key = gr.Textbox(label="API Key", type="password", placeholder="sk-…")
                            pe_llm_url = gr.Textbox(
                                label="Base URL",
                                placeholder="https://tokenhub.tencentmaas.com/v1",
                                info="填 API 根地址，不要带 /chat/completions",
                            )
                            pe_llm_model = gr.Textbox(label="模型名", placeholder="hy3")

                        with gr.Accordion("高级参数", open=False):
                            in_nfe = gr.Slider(
                                4,
                                64,
                                value=initial_sampling["nfe"],
                                step=1,
                                label="NFE 步数",
                                interactive=initial_sampling["interactive"],
                            )
                            in_cfg = gr.Slider(
                                0.0,
                                5.0,
                                value=initial_sampling["cfg"],
                                step=0.1,
                                label="CFG 强度",
                                interactive=initial_sampling["interactive"],
                            )
                            in_seed = gr.Number(label="随机种子（可留空）", value=42, precision=0)
                            gr.Markdown(SAMPLING_NOTE, elem_classes="auk-hint")

                    # 右侧宽列：主输入 + 结果，scale=2
                    with gr.Column(scale=2):
                        in_instr = gr.TextArea(
                            label="指令（Instruction）",
                            placeholder="用自然语言描述要做什么，例如：把「but accepting」替换成「and living well」。",
                            lines=6,
                            max_lines=12,
                        )
                        with gr.Accordion("时长控制", open=False):
                            in_secs = gr.Slider(
                                0,
                                30,
                                value=0,
                                step=0.5,
                                label="目标时长（秒）",
                            )
                            gr.Markdown(
                                "0 = 自动（有参考音频时与源等长；无参考音频时必须指定）。",
                                elem_classes="auk-hint",
                            )
                        out_audio = gr.Audio(
                            label="生成结果",
                            buttons=["download"],
                            elem_classes=["compact-audio"],
                        )

                # 内部状态：Prompt Enhancer 的回填值，界面上不展示
                in_ref_text = gr.State("")
                in_gen_text = gr.State("")

                # 状态行放 Row 外，长文本不挤占布局
                gen_status = gr.Markdown("⏸ 等待生成")
                out_info = gr.Markdown("")
                # =================== 示例（本页底部） =================== #
                gr.Markdown("---")
                gr.Markdown(
                    "## 🧩 示例\n\n"
                    "选中任意一条示例即可把参数填到上面的面板，再点「🚀 开始生成」。\n\n"
                    "- 📖 **README 示例**：直接解析 `README.md` 里的命令，文档更新这里自动同步；\n"
                    "- 音频格为空表示**示例音频缺失**，请自行上传一段音频后再生成；\n"
                    "- 提示词在表格里**完整换行显示**，不会省略号截断。"
                )

                readme_groups = readme_example_rows()
                builtin_groups = builtin_example_rows()
                total_readme = sum(len(rows) for rows in readme_groups.values())
                total_builtin = sum(len(v) for tasks in builtin_groups.values() for v in tasks.values())

                with gr.Tabs():
                    # ---------- README 示例（主）----------
                    with gr.Tab(f"📖 README 示例（{total_readme}）"):
                        if not readme_groups:
                            gr.Markdown("⚠️ 未能从 `README.md` 解析到示例命令。")
                        else:
                            with gr.Tabs():
                                for heading, rows in readme_groups.items():
                                    if not rows:
                                        continue
                                    with gr.Tab(heading):
                                        dataset = gr.Dataset(
                                            components=[
                                                gr.Audio(label="音频", render=False),
                                                gr.Textbox(label="指令", render=False),
                                            ],
                                            samples=_example_samples(rows),
                                            headers=["音频", "指令（完整显示）"],
                                            type="index",
                                            layout="table",
                                            samples_per_page=10,
                                            show_label=False,
                                            elem_classes=["auk-examples-table"],
                                        )
                                        dataset.select(
                                            lambda index, r=rows: load_demo_example(index, r),
                                            inputs=dataset,
                                            outputs=[in_audio, in_instr, in_secs, use_pe],
                                            queue=False,
                                            api_visibility="private",
                                            show_progress="hidden",
                                        )
                                        # 单独的纯 JS 步骤负责回到页面顶部。
                                        # 不能给上面那个 select 传 js——那会让后端回填失效。
                                        dataset.select(
                                            None,
                                            js=_SCROLL_TOP_JS,
                                            queue=False,
                                            api_visibility="private",
                                            show_progress="hidden",
                                        )

                    # ---------- 内置示例 ----------
                    with gr.Tab(f"🧪 更多示例（{total_builtin}）"):
                        with gr.Tabs():
                            for category, tasks in builtin_groups.items():
                                with gr.Tab(category):
                                    with gr.Tabs():
                                        for task, rows in tasks.items():
                                            if not rows:
                                                continue
                                            with gr.Tab(task):
                                                is_instruct_tts = task == "Instruct TTS"
                                                if is_instruct_tts:
                                                    components = [gr.Textbox(label="指令", render=False)]
                                                    headers = ["指令（完整显示）"]
                                                else:
                                                    components = [
                                                        gr.Audio(label="音频", render=False),
                                                        gr.Textbox(label="指令", render=False),
                                                    ]
                                                    headers = ["音频", "指令（完整显示）"]
                                                dataset = gr.Dataset(
                                                    components=components,
                                                    samples=_example_samples(rows, only_instruction=is_instruct_tts),
                                                    headers=headers,
                                                    type="index",
                                                    layout="table",
                                                    samples_per_page=8,
                                                    show_label=False,
                                                    elem_classes=["auk-examples-table"],
                                                )
                                                dataset.select(
                                                    lambda index, r=rows: load_demo_example(index, r),
                                                    inputs=dataset,
                                                    outputs=[in_audio, in_instr, in_secs, use_pe],
                                                    queue=False,
                                                    api_visibility="private",
                                                    show_progress="hidden",
                                                )
                                                dataset.select(
                                                    None,
                                                    js=_SCROLL_TOP_JS,
                                                    queue=False,
                                                    api_visibility="private",
                                                    show_progress="hidden",
                                                )

            # ======================= 历史记录 ======================= #
            with gr.Tab("🗂 历史记录"):
                gr.Markdown(f"这里列出 `{HISTORY_DIR}` 下的产物。进入时不加载音频，**选中某段才会加载**。")
                with gr.Row():
                    hist_refresh_btn = gr.Button("🔄 刷新列表", scale=1)
                    hist_info = gr.Markdown("点「刷新列表」加载历史。")
                with gr.Row():
                    with gr.Column(scale=1):
                        hist_list = gr.Radio(choices=[], label="历史列表（选中即播放）", elem_id="history_list")
                    with gr.Column(scale=1):
                        hist_audio = gr.Audio(label="试听 / 下载", autoplay=True, buttons=["download"])
                with gr.Row():
                    hist_del_confirm = gr.Checkbox(label="我确认删除全部产物", value=False, scale=1)
                    hist_del_btn = gr.Button("🗑 删除所有产物", variant="stop", scale=1)

            # ======================= 示例 ======================= #
            # ======================= 模型管理 ======================= #
            with gr.Tab("🧰 模型管理"):
                gr.Markdown(
                    "WebUI **默认懒加载**：启动时不占显存，第一次生成时才载入模型。\n\n"
                    "在这里可以**手动切换模型**——切换时会自动卸载旧模型、释放显存"
                    "（单卡 23 GiB 容不下两个变体同时常驻）。"
                )
                engine_status = gr.Markdown(ENGINES.status_markdown())
                with gr.Row():
                    engine_switch_dd = gr.Dropdown(
                        choices=list_variants() or [PLACEHOLDER],
                        value=(list_variants()[0] if list_variants() else PLACEHOLDER),
                        label="切换到",
                        scale=6,
                    )
                    engine_switch_btn = gr.Button("🔀 切换到此模型", variant="primary", scale=2)
                    engine_refresh_btn = gr.Button("🔄 刷新状态", scale=1)
                with gr.Row():
                    engine_unload_dd = gr.Dropdown(
                        choices=ENGINES.loaded() or [PLACEHOLDER],
                        value=(ENGINES.loaded()[0] if ENGINES.loaded() else PLACEHOLDER),
                        label="卸载",
                        scale=6,
                    )
                    engine_unload_btn = gr.Button(
                        "🧹 卸载此模型 / 释放显存",
                        variant="secondary",
                        interactive=bool(ENGINES.loaded()),
                        scale=2,
                    )
                    engine_unload_all_btn = gr.Button(
                        "🗑 卸载全部",
                        variant="stop",
                        interactive=bool(ENGINES.loaded()),
                        scale=1,
                    )
                engine_msg = gr.Markdown("")

            # ======================= 使用说明 ======================= #
            with gr.Tab("📖 使用说明"):
                gr.Markdown(HELP_MARKDOWN)

        # ══════════════════════════════════════════════════════════════
        # §EVENTS  事件绑定（模式：做事 → .then() 刷新）
        # ══════════════════════════════════════════════════════════════

        # 主 CTA：先跑任务，再刷新历史列表 —— 两步分离
        gen_btn.click(
            run_task,
            inputs=[
                variant_dd,
                in_audio,
                in_instr,
                in_secs,
                in_ref_text,
                in_gen_text,
                in_nfe,
                in_cfg,
                in_seed,
                use_pe,
                pe_llm_key,
                pe_llm_url,
                pe_llm_model,
            ],
            outputs=[gen_status, out_audio, out_info],
        ).then(refresh_history, None, [hist_list, hist_info])

        # 勾选/填写 PE 凭据时即时反馈是否可用，不用等到点生成
        pe_llm_key.change(
            check_pe, [use_pe, pe_llm_key, pe_llm_url, pe_llm_model], [pe_hint, use_pe], queue=False, api_visibility="private"
        )
        pe_llm_url.change(
            check_pe, [use_pe, pe_llm_key, pe_llm_url, pe_llm_model], [pe_hint, use_pe], queue=False, api_visibility="private"
        )
        pe_llm_model.change(
            check_pe, [use_pe, pe_llm_key, pe_llm_url, pe_llm_model], [pe_hint, use_pe], queue=False, api_visibility="private"
        )
        use_pe.change(
            check_pe, [use_pe, pe_llm_key, pe_llm_url, pe_llm_model], [pe_hint, use_pe], queue=False, api_visibility="private"
        )

        # 历史：选中才加载；删除前必须勾确认框
        hist_refresh_btn.click(refresh_history, None, [hist_list, hist_info])
        hist_list.change(pick_history, [hist_list], [hist_audio, hist_info])
        hist_del_btn.click(delete_history, [hist_del_confirm], [hist_list, hist_audio, hist_info])

        # 模型管理：手动切换 / 刷新状态 / 卸载
        engine_switch_btn.click(switch_engine, [engine_switch_dd], [engine_status, engine_msg]).then(
            refresh_engines, None, [engine_status, engine_unload_btn]
        )
        engine_refresh_btn.click(refresh_engines, None, [engine_status, engine_unload_btn])
        engine_unload_btn.click(
            unload_one_engine,
            [engine_unload_dd],
            [engine_status, engine_unload_btn, engine_msg],
        )
        engine_unload_all_btn.click(unload_all_engines, None, [engine_status, engine_unload_btn, engine_msg])

        # 主功能页切换模型下拉：同步卸载其它变体释放显存（但不预加载，保持懒加载）
        variant_dd.change(
            update_sampling_controls,
            inputs=variant_dd,
            outputs=[in_nfe, in_cfg],
            queue=False,
            api_visibility="private",
        ).then(
            release_other_engines,
            [variant_dd],
            [engine_status, engine_msg],
            queue=False,
            api_visibility="private",
        )

        # ---- 页面加载时回填状态：刷新不丢状态 ----
        # show_progress 必须设为 hidden：否则 load 事件的 status-tracker 会嵌在
        # hist_list 里，事件完成后仍不消失，表现为「历史记录」Tab 一直转圈。
        demo.load(
            refresh_history,
            None,
            [hist_list, hist_info],
            show_progress="hidden",
        )

        return demo


# ══════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════


def main() -> None:
    # 允许 --ckpt-dir / --output-dir 覆盖默认位置（须在任何使用之前声明）
    global CKPT_DIR, OUTPUT_DIR, HISTORY_DIR

    parser = argparse.ArgumentParser(description="AuK 语音生成与编辑工作台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；0.0.0.0 允许外部设备访问")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--ckpt-dir", default=str(CKPT_DIR), help="模型权重根目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="产物输出目录")
    parser.add_argument("--preload", action="store_true", help="启动时立即加载模型（放弃懒加载）")
    parser.add_argument("--share", action="store_true", help="生成临时公网链接（仅演示用）")
    args = parser.parse_args()

    CKPT_DIR = Path(args.ckpt_dir)
    OUTPUT_DIR = Path(args.output_dir)
    HISTORY_DIR = OUTPUT_DIR

    if args.preload:
        for variant in list_variants():
            ENGINES.get(variant)

    demo = build_demo()
    # 单卡 24 GiB：并发设 1，避免两个请求同时推理把显存撑爆
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Ocean(),
        css=CSS,
        allowed_paths=[os.getcwd(), str(OUTPUT_DIR)],
    )


if __name__ == "__main__":
    main()
