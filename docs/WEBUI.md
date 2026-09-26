# AuK WebUI（二次开发）

基于 Gradio Blocks 的 AuK 推理工作台，相对官方 `auk-gradio` 增加了**懒加载**、**统一产物目录**和**默认 CPU offload**。

## 启动

```bash
# 一键启动（自动清端口 + 清显存 + 允许外部访问）
./start_app.sh 7860

# 或手动启动
export HF_ENDPOINT=https://hf-mirror.com
python -m webui.app --host 0.0.0.0 --port 7860
```

打开 `http://<服务器IP>:7860`。

启动参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | `0.0.0.0` 才能让外部设备访问 |
| `--port` | `7860` | 监听端口 |
| `--ckpt-dir` | `<repo>/ckpts` | 模型权重根目录 |
| `--output-dir` | `<repo>/outputs` | 产物输出目录 |
| `--preload` | 关 | 启动即加载模型（放弃懒加载） |
| `--share` | 关 | 生成临时公网链接，仅演示用 |

环境变量：`AUK_CKPT_DIR`、`AUK_OUTPUT_DIR`、`AUK_PRELOAD=1`、`HF_ENDPOINT`。

## 下载模型

模型放在仓库的 `ckpts/` 下（与官方布局一致，`config.yaml` 里的相对路径无需改写）：

```
ckpts/
├── AuK/{auk_base.safetensors, vae.safetensors, config.yaml}
└── Qwen2.5-Omni-3B/...
```

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py                # download + verify
python scripts/download_models.py --verify-only  # re-verify only
python scripts/download_models.py --variant auk  # only the DiT/VAE release
```

模型按需分批下载，`--variant` 可选 `auk`（Base）、`flash`（AuK-Flash）、`qwen`（编码器）、`all`：

```bash
# 空间紧张时只补 Flash，并把磁盘预留门槛从默认 3 GiB 降到 1.5 GiB
python scripts/download_models.py --variant flash --reserve 1.5
```

两个优化：

- **VAE 硬链接复用**：`AuK` 与 `AuK-Flash` 的 `vae.safetensors` 是同一个文件（SHA-256 一致），
  脚本会把已校验过的那一份硬链接过去，**不占额外磁盘**（实测省 608 MiB）。
- **断点续传计入预检**：中断后重跑，`.part` 里已下载的字节算作已到位，
  不会把整份文件当成全新下载而多要一倍空间。

manifest 采用**合并写入**，分批下载不会冲掉先前已登记的文件；结果写到
`ckpts/.checksums.sha256`（`sha256sum -c` 兼容），可用 `--verify-only` 随时复查。

> 若要使用 **AuK-Flash**，把 `ckpts/AuK-Flash/` 也下下来即可，WebUI 的模型下拉会自动多出一项。

## 产物命名

所有生成结果统一落到 `outputs/`，文件名格式为：

```
outputs_YYYYMMDD_HHMMSS.wav        # 例：outputs_20260926_143512.wav
outputs_YYYYMMDD_HHMMSS_001.wav    # 同一秒内重复生成时自动追加序号
```

文件名由 `touch(exist_ok=False)` 原子占名，即使 Gradio 队列并发也不会重名；生成失败时占名的空文件会被自动清理。

## 懒加载

启动 WebUI 时**不加载任何模型、不占显存**。第一次点「🚀 开始生成」时才构造 `AukInfer`
（约 1–2 分钟），之后同一变体常驻复用。

在「🧰 模型管理」页可以查看已加载变体与显存占用，并随时「卸载全部模型 / 释放显存」。

## 切换模型与显存释放

**单模型常驻策略**：单卡 23.5 GiB 容不下两个变体同时推理（Base / Flash 峰值各约 10 GiB），
因此无论从哪个页面切换模型，都**只会保留一个变体在内存**，旧模型立即被卸载。

| 入口 | 行为 |
|---|---|
| 🎧 主功能页 模型下拉 | 切过去立即卸载其它变体释放显存；新变体仍等到首次生成才载入（保持懒加载） |
| 🧰 模型管理页 「🔀 切换到此模型」 | 卸载其它变体 + 立即加载选中变体（常驻，可直接生成） |
| 🧰 模型管理页 「🧹 卸载此模型」 | 只卸载指定变体 |
| 🧰 模型管理页 「🗑 卸载全部」 | 全部卸载，显存归零 |

> CPU offload 开启时，模型**常驻**显存只有约 1.4 GiB（VAE + 层融合参数），
> 真正的大头是**推理峰值**约 10 GiB。所以「释放显存」的实际意义是避免两个 10 GiB 峰值叠加爆显存。

## Prompt Enhancer

PE 靠 **LLM** 识别任务、抽取参数、生成规范指令——**没有本地兜底路径**，所以必须配一个 OpenAI 兼容端点，否则不可用。

两种配法：

1. **界面直接填**（即刻生效，不写盘）：展开「🎧 主功能」页的 **Prompt Enhancer 设置**折叠面板，填
   `API Key` / `Base URL` / `模型名`。勾选「使用 Prompt Enhancer」后会即时提示是否就绪。
2. **环境变量**（长期生效）：`cp .env.example .env` 填好后重启。留空时界面输入优先读这三项：
   `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`。

> `LLM_BASE_URL` 填 API 根地址，**不要**带 `/chat/completions`。

不上 LLM 也能用：取消勾选「使用 Prompt Enhancer」，所有任务都可以直接写指令完成，PE 只是可选的增强步骤。

音频转写（ASR）优先用腾讯云录制文件识别；无凭据或云端失败时，自动下载 `SenseVoiceSmall` 到 CPU 兜底
（首次约 1 GiB）。

## 页面结构

| Tab | 作用 |
|---|---|
| 🎧 主功能 | 参考音频 + 指令 → 生成；**示例列表在本页底部**；高级参数（NFE / CFG / 种子）与时长控制收在折叠面板里 |
| 🗂 历史记录 | 列出 `outputs/` 产物，进入不加载音频，选中才播放；删除需勾选确认框 |
| 🧰 模型管理 | 手动切换模型、按变体卸载、显存占用、一键全卸 |
| 📖 使用说明 | 面向新用户的内置指引：三步上手、页面导览、指令写法、10 条 FAQ（内容见 `webui/app.py` 的 `HELP_MARKDOWN`） |

### 关于「🎧 主功能」页底部的示例区

页面顶部有 `id="top"` 锚点：地址栏访问 `/#top` 会直接回到顶部；点击任意示例后也会自动平滑滚回顶部（示例在页底，回填的面板在上方，所以这一步是必要的）。

示例区紧跟在生成结果下方，选中任意一条即可把参数填到本页上方的面板，再点「🚀 开始生成」。分两部分：

- **📖 README 示例** — `webui/app.py` 在页面加载时**直接解析 `README.md` 里的命令**，按小节标题归类。文档里新增/修改示例，刷新页面即可看到，无需改代码。
- **🧪 更多示例** — 内置的 `DEMO_EXAMPLE_GROUPS`（来自公开 demo 样例集），覆盖约 56 条各任务指令。

规则：

- **音频格为空 = 示例音频缺失**（例如 `assets/after_pe/` 是 Prompt Enhancer 的运行时产物，仓库里没有）。此时请自行上传一段音频再生成；不会因为没有音频就把示例藏掉。
- **提示词完整换行显示**，不省略号截断（CSS 见 `webui/app.py` 的 `.auk-examples-table`）。表格内最多显示 340px 高度，超出可在单元格内滚动。
- PE 勾选框**永不自动勾上**：即便示例本身是 PE 命令，也留给你手动决定，避免没配 LLM 时一点示例就直接报错。

## 端口被占用怎么办

`start_app.sh` 会 **无交互** 终止占用目标端口的旧进程（只杀该 PID，不碰其它进程），并用
`fuser $port/tcp` 轮询最多 3 秒确认端口真正释放后才启动主程序。启动前还会清理其它
占用显存的进程，保证本次运行独占 GPU。
