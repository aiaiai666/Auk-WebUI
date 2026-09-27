# HANDOFF · AuK WebUI 排障与 Prompt Enhancer 本地化

> 写给**完全没有上下文**的新会话。先读 §0 和 §1，再看 §7（踩坑清单）。
> 记录的是 2026-09-27 一次排障会话：全部改动**已提交并推送到 `main`**，本文档保留为
> 「做了什么 / 为什么这么做 / 哪些坑别再踩」的依据，不是实时状态报告。
> §3 逐条给出症状→根因→修法→验证，§7 是踩坑清单，§8 是可复现的验证命令。

---

## 0. 一句话总结

用户报了四个依次暴露的问题——① 生成不出音频、② 勾选 Prompt Enhancer 报错、③ 勾选 PE 后报临时文件不存在、
④ 勾选 PE 后「耳语转正常」示例生成**静音**；后续还要求把 ASR 权重迁入 `ckpts/`。
最终挖出 **14 个独立缺陷**（含 4 个既有 bug、1 个本次自己引入的），全部修完并端到端验证通过。
核心结论：**PE 需要的 LLM 和 `ckpts/Qwen2.5-Omni-3B` 毫无关系**——后者是 AuK 的 text encoder，不是对话端点。


---

## 1. 任务背景与目标

| 项 | 内容 |
|---|---|
| 仓库 | AuK-WebUI，分支 `main`，remote `origin` = `https://github.com/aiaiai666/Auk-WebUI` |
| 基线 commit | `a5796ed chore: track AGENTS.md in version control` |
| 落地范围 | `a5796ed..6d7f8f4` 共 12 个 commit（含本文档），全部已推送 |
| 用户诉求 1 | 「生成音频报」→ 排查为什么 `outputs/` 里没有产物 |
| 用户诉求 2 | 「Qwen2.5-Omni-3B 已下载，勾选 PE 还是报错，修复」 |
| 用户诉求 3 | 「勾选 PE + 耳语转正常示例 → 生成的音频没有声音」 |
| 用户诉求 4 | 把 `~/.cache/modelscope` 里的 ASR 权重移进 `ckpts/`，方便项目迁移 |
| 用户明确选择 | 用**本地 Qwen 起 OpenAI 兼容服务**（而非填云端 key），所以 §3.4 的方案是用户拍板的 |
| 交付形态 | 代码修复 + 文档更新，已提交并推送 |

### 环境事实（复现时先核对这些）

```
Python 3.13.11
torch 2.7.1+cu126 | torchaudio 2.7.1+cu126 | transformers 4.52.0
gradio 6.17.3 | openai 3.19.2 | fastapi 0.141.1 | uvicorn 0.54.0
GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
RAM: 56 GiB

装好的相关包: audioread 3.1.0 | funasr 1.4.16 | modelscope 1.40.1 | qwen-omni-utils 0.0.9
模型: ckpts/AuK/{auk_base.safetensors, vae.safetensors, config.yaml}
      ckpts/AuK-Flash/{auk_flash.safetensors, vae.safetensors, config.yaml}
      ckpts/Qwen2.5-Omni-3B/  (3 shards, ~11.4 GB)
ASR 缓存: ~/.cache/modelscope/models/iic--SenseVoiceSmall/  (已下载, 936 MB)
```

**运行位置约束**：`config.yaml` 里 `ckpts/Qwen2.5-Omni-3B` 是相对 CWD 解析的，
所以**任何 python 命令都必须在 the repository root 下执行**。

---

## 2. 症状 → 根因速查表

| # | 症状 | 根因 | 位置 | 状态 |
|---|---|---|---|---|
| 1 | 生成必失败，`outputs/` 永远空 | `qwen-omni-utils` 顶层 `import audioread`，但其元数据没声明这个依赖 | 环境缺包 | 已修 |
| 2 | 生成失败但磁盘无痕迹 | 异常被兜底分支吞掉后 `unlink` 掉占名空文件 | `webui/app.py` `run_task` | 既有设计 |
| 3 | 勾选 PE 必报「未配置 LLM」，即使 `.env` 配好 | `_prepare_with_pe` 只看传入的页面字段，**从不回落 env** | `webui/app.py` `_prepare_with_pe` | 已修 |
| 4 | ASR 静默失效（`缺少 FunASR 依赖`） | `funasr`/`modelscope` 未安装 | 环境 | 已修 |
| 5 | PE 无 LLM 可用 | PE 强制要 OpenAI 兼容端点，无兜底 | 设计 | 已加本地端点 |
| 6 | 本地 3B 偶发 `operation_subtype: null` → PE 报「缺少或包含非法 operation_subtype」 | `_classify` 单次调用、无校验修复 | `src/auk/infer/pe.py` | 已修（有界重试） |
| 7 | **勾选 PE 报 `Failed to open the input "/tmp/auk_pe_level_*.wav"`** | `_prepare_with_pe` 的 `finally` 在调用方使用前就 `cleanup()` 删掉了 PE 交出的临时音频 | `webui/app.py` | 已修 |
| 8 | 跑一会儿 CUDA OOM | 本地 LLM 服务没在生成后 `empty_cache()`，reserved 稳态就停在 19.29 GiB | 新脚本 | 已修（稳态 8.5 GiB） |
| 9 | 重启 WebUI 会杀掉自己的 PE 依赖 | `start_app.sh` 不加载 `.env`，且 `free_gpu_memory` 杀所有非本树 GPU 进程 | `start_app.sh` | 已修 |
| 10 | `[: 0.1: 需要整数表达式`，等待逻辑被静默跳过 | `waited` 被 awk 累成浮点，却用 `-lt` 整数比较 | `start_app.sh` | 已修（既有 bug） |
| 11 | **勾选 PE + 「耳语转正常」示例 → 输出静音（−63 dBFS）** | `to_normal_target_rms` 配成了 **0.000708**，比耳语目标 `0.0064` 还小 19 dB，把参考音频**调成了静音** | `pe.config.yaml` | 已修（0.05 + 代码护栏） |
| 12 | `python scripts/download_models.py` 裸跑必中止「空间不足」 | `--reserve` 默认值是字节，`main()` 又按 GiB 乘一次 `1024**3`，3 GiB 变成 3072 PiB | `scripts/download_models.py` | 已修（既有 bug） |
| 13 | 全新安装会失败 | `audioread>=3.3` 不可满足，最新版只有 3.1.0 | `pyproject.toml` | 已修（本次自己引入） |
| 14 | 仓库换个位置就丢 ASR 能力 | ASR 权重只存在于 `~/.cache/modelscope` | `pe.py` | 已修（迁入 `ckpts/SenseVoiceSmall`） |


---

## 3. 已完成的改动（逐条：症状 → 根因 → 修法 → 验证）

### 3.1 补 `audioread` 依赖

**根因**：`qwen_omni_utils/v2_5/audio_process.py:4` 顶层 `import audioread`，
但 `pip show qwen-omni-utils` 的 `Requires:` 只有 `av, librosa, packaging, pillow, requests`。
按文档装完依赖后首次生成必炸。调用链：
`AukInfer.generate` → `_run` → `CFMEdit.build_cond_inputs` → `import qwen_omni_utils` → `import audioread`。

**修法**：装包 + 补进 `pyproject.toml` 的 **core `dependencies`**（不是 extras——
因为不装 extras 的纯 `pip install -e .` 走同一条 import 链一样会崩）。

```bash
pip install audioread          # 3.1.0
```

`pyproject.toml` 里加了带注释的 `audioread>=3.3`，说明为什么它必须在 core。

### 3.2 修复 PE 的 env 回落不可达（**用户侧真正的 bug**）

**症状**：`.env` 配得完全正确，`os.environ` 里 `LLM_API_KEY/LLM_BASE_URL/LLM_MODEL_NAME` 都在，
但勾选 PE 仍然报「未配置」。

**根因**：`_prepare_with_pe` 里是

```python
if not (llm_api_key and llm_base_url and llm_model):   # 只看入参
    raise gr.Error("**未配置 Prompt Enhancer 所需的 LLM。**…")
```

这三个入参来自页面三个输入框，正常用户都是留空（靠 `.env`）。
所以**它自己错误文案里推荐的「方式一 · 环境变量」路径根本不可达**——文案在教用户一条死路。

**修法**：复用文件里已有的 `llm_ready()`（`app.py` 里本来就实现了 env 回落语义）：

```python
if not llm_ready(llm_api_key, llm_base_url, llm_model):
```

**验证**：`_prepare_with_pe` 在 env 已设、页面留空时正常返回。

### 3.3 补 ASR 依赖

`funasr>=1.3,<2` / `modelscope>=1.20,<2` 在 `pyproject.toml` 的 `gradio` extra 里**有声明**，
但环境里没装（说明当初没按 `pip install -e ".[gradio]"` 装全）。PE 的 ASR 会静默降级：

```
asr = model=SenseVoiceSmallASR err='缺少 FunASR 依赖，请重新安装 AuK' text=''
```

**修法**：

```bash
pip install "funasr>=1.3,<2" "modelscope>=1.20,<2"
# 装完必须核对 torch 没被带偏：
python3 -c "import torch;print(torch.__version__)"   # 期望 2.7.1+cu126
```

装完 `torch 2.7.1+cu126` 未变，ASR 立刻出真实转写（`lang=en`）。

### 3.4 新增 `scripts/local_llm_server.py`（用户拍板的方案）

把已下载的 `ckpts/Qwen2.5-Omni-3B` 包成 OpenAI 兼容 `/v1/chat/completions`。

**为什么可行**：`pe._call_llm` 只发**纯文本**（`messages: list[dict[str, str]]`，`pe.py` 内），
从不发音频。而 Qwen2.5-Omni 的 Thinker 本身就是指令微调过的对话模型（我最初误以为是 base 模型，
实测分类正确后撤回该顾虑）。所以只需保留 Thinker 文本路径，`visual` / `talker` 在加载时删掉
——和 `AukInfer` 的做法一致。

**关键实现点（每一条都是踩出来的）**：

1. **必须显式传 `eos_token_id`**。该 checkpoint 的 `generation_config.eos_token_id` 是 `None`，
   `generate()` 永不停止，会在答案后继续吐 `Human:/Assistant:` 噪声。
   而 `pe._extract_json` 的兜底正则 `\{.*\}`（DOTALL、**贪婪**）会跨段匹配后 `json.loads` 失败。
   chat template 用 `<|im_end|>`(151645) 收尾，所以传 `eos_token_id=[151645, 151643]`。
   实测：不传时 200 token 的回复里只有前 6 个 token 是有效 JSON。
2. **每次生成后 `torch.cuda.empty_cache()`**。caching allocator 只涨不缩，
   实测 7.6 GiB 的模型 reserved 冲到 **19.29 GiB**，把 AuK（峰值约 10.9 GiB）挤到 OOM。
   加了之后稳定在 **8.4–8.5 GiB**。
3. **`--max-new-tokens` 默认 1024**（约 45s）。`pe` 单次调用超时是 120s
   （`pe.config.yaml` 的 `api.llm.timeout_sec`），而 `pe` 会请求 4096——不限就会超时。
4. **`max_tokens` 会被 clamp**，不无条件信任客户端。
5. asyncio lock 串行化生成，并发请求排队而不是抢显存。
6. 端点**不鉴权**，默认只绑 `127.0.0.1`，`--host` 可改但要在文档里写清风险。

### 3.5 `start_app.sh` 改造

新增三件事（原有「三件事」变「五件事」）：

1. **加载 `.env`**（`set -a; source ./.env; set +a`）。之前**根本不读**，
   所以 `.env` 里的 PE 凭据永远不生效。顺带把 banner 的「PE 端点」显示修对
   （`load_env` 必须排在 banner 之前）。
2. **自动拉起本地 LLM**：`.env` 的 `LLM_BASE_URL` 指向 loopback 且 `/health` 不通时，
   `setsid nohup` 起服务并轮询 `/health`（上限 `LLM_READY_TIMEOUT=180`）。
   关键：**用 `setsid` + `disown`**，否则脚本退出时子进程被一起回收
   （踩过：第一次用 `nohup ... &` 起服务，下一条命令就变成 `APIConnectionError`）。
3. **显存清理豁免本地 LLM**：按**端口**识别（`is_local_llm_pid`），
   否则每次重启 WebUI 都会杀掉自己的 PE 依赖。SIGKILL 补刀那轮同样要豁免。
   注意：端口解析（`resolve_llm_endpoint`）必须发生在 `free_gpu_memory` **之前**，
   否则豁免判断拿到空值。
4. 顺手修了 §2 #10 的浮点/整数比较（`PORT_WAIT_POLLS` / `GPU_WAIT_POLLS` 整数轮询计数）。

### 3.6 `pe.py`：分类校验 + 有界重试

**症状**：`把「but accepting」替换成「and living well」`（**直角引号**）→ PE 报
`content_edit 缺少或包含非法 operation_subtype: None`。
用直引号 `"but accepting"` 则正常。3 次重跑结果**完全一致**，是确定性的，不是随机。

**根因**：`_classify` 单次调用、无任何校验修复。小模型漏填 `operation_subtype` 就整单失败。
换强模型（官方默认 hy3 / deepseek-v4-flash）不暴露，但任何小模型都会。

**修法**：
- `_classify` 拆出 `_parse_classified()`，把「合法 task_type + 合法 operation_subtype」
  的校验前移到解析阶段；
- 校验失败时把**错误原文回喂**让模型自己改（`_classify_repair_hint`），
  最多 `api.llm.classify_max_attempts` 次（`pe.config.yaml` 新增，默认 3，设 1 = 关闭重试）；
- 返回值从 2 元组变 3 元组，多带一份 `llm_calls` 以便重试记录不丢。

**验证**：第 1 次 null → 回喂 → 第 2 次 `replace` → 通过。
`llm_calls` 正确显示 `[('classify',336), ('classify',341)]` 两次调用。

### 3.7 `webui/app.py`：PE 临时音频的所有权（**用户最后报的那个错**）

**症状**：勾选 PE 报
`RuntimeError: Failed to open the input "/tmp/auk_pe_level__i1swarn.wav" (No such file or directory)`；
不勾选完全正常。

**根因**：部分 PE 任务会先写临时 wav 再交给生成端：

```
pe._prepare_audio()  →  VAD 裁剪 / 响度归一化
                       →  返回 (临时路径, cleanup_paths=[临时路径, ...])
                       →  PromptEnhancerOutput.audio = 临时路径
webui._prepare_with_pe()  →  try: return (..., prepared.audio, ...)
                          →  finally: prepared.cleanup()   ← return 求值后立刻删掉它
run_task()  →  engine.generate(audio=<已被删除的路径>)  →  炸
```

**对照官方实现**：`src/auk/infer/infer_gradio.py` 是**对的**——
`run_generate()` 和 `cleanup()` 在同一个 `try` 内，清理发生在生成**之后**。
`webui/app.py` 把 prepare 与 generate 拆成两个函数，却把清理留在了 prepare 里。
**这是 `webui/app.py` 独有的 bug，`auk-gradio` 不受影响。**

**修法**：`_prepare_with_pe` 把 `PromptEnhancerOutput` 作为**第 7 个返回值**交出所有权，
清理移到 `run_task` 的 `finally`（生成之后）；`handed_off` 标志保证失败路径仍就地清理、不泄漏临时文件。

**验证**：
```
whisper_edit/to_normal → /tmp/auk_pe_level_wkl8cj1l.wav
  _prepare_with_pe 返回后 exists=True（修复前必然 False）
  prepared.cleanup() 后 exists=False
  端到端：50.4s 生成 outputs_20260927_055129.wav（8.6s 音频）
  /tmp 无 auk_pe_* 残留
```

### 3.8 耳语「转正常」生成静音（配置常量方向反了）

**症状**：WebUI 示例区「提示词增强器」第一个例子
（`assets/demo-input-audio/whisper/wh-w2n-zh-input.wav` +
`Convert this whisper into normal speech…`），勾选 PE → 生成的音频**完全没声音**。
不勾选 PE 正常。

**排查链（关键：不勾选正常 → 病灶一定在 PE 改动了的东西上）**：

1. 量化产物，确认是静音而非"小声"：
   ```
   outputs_…055129.wav   rms=0.000644  peak=0.0046   ≈ -63.8 dBFS
   原始输入               rms=0.049221  peak=0.3641   ≈ -26.2 dBFS
   ```
2. 关掉 PE，同音频同指令跑 6 次（两个措辞 × seed 42/42/7）→ 全部 rms 0.041–0.055，**模型本身没问题**。
3. 于是去查 PE 到底改了什么 → `prepared.audio`：
   ```
   PE 交出的音频   sr=40000 ch=1 rms=0.000708 peak=0.0052   ← 本身就是静音
   ```
   **模型拿到静音参考，输出静音。** 不是模型问题，是喂进去的东西坏了。
4. 定位到 `_prepare_audio` 的 `to_normal` 分支 → `_normalize_audio_level` → 配置：
   ```yaml
   whisper:
     target_rms: 0.0064            # 耳语目标，-43.9 dBFS
     to_normal_target_rms: 0.000707945784384138   # -63.0 dBFS ← 比耳语还小 19 dB
   ```
   实测 PE 输出 rms=0.000708 **正好等于**该目标值，坐实是配置驱动的。
5. **不变量**：耳语是音量区间的安静端，所以「转正常」的目标**必须比耳语目标更响**。
   原值方向完全反了——它在把耳语**调得更小**。

**修法（两处）**：

- `pe.config.yaml`：`to_normal_target_rms: 0.000707945784384138` → `0.05`
  （-26 dBFS，与 AuK 对该音频的正常输出电平一致），并加注释写明上面那条不变量。
- `pe.py::_prepare_audio` 的 `to_normal` 分支改成
  `target_rms=max(WHISPER_TO_NORMAL_TARGET_RMS, WHISPER_TARGET_RMS)`
  —— 让不变量由**代码**保证而不是只靠注释。改后即使配置再被写错也不会静默产出静音。

**验证**：

```
PE 参考音频   0.000708 (-63.0 dBFS) → 0.050000 (-26.0 dBFS)
输出          0.000644 (-63.8 dBFS) → 0.046455 (-26.7 dBFS)   与关 PE 的基线一致
反向 to_whisper 分支未回归：rms=0.0064（= target_rms），peak 0.0473
端到端 run_task：51.2s → outputs_20260927_061557.wav，8.58s，-26.7 dBFS
```

### 3.9 `download_models.py` 的 `--reserve` 单位错乱（阻断性既有 bug）

**症状**：`python scripts/download_models.py`（README 与 AGENTS.md 记载的主流程，**不带任何参数**）
必然中止：

```
Required : 0.0 B new / 65.7 GiB free (reserve 3072.0 PiB)
[ERROR] not enough free space in ckpts; aborting before downloading.
```

**根因**：`--reserve` 的 argparse 默认值传的是**字节**，而 `main()` 又按 **GiB** 乘了一次 `1024**3`：

```python
RESERVE_BYTES = 3 * 1024**3                     # 默认值：字节
...
reserve = int(args.reserve * 1024**3)            # 又乘 1024³ → 3 GiB 变成 3072 PiB
```

结果任何不带 `--reserve` 的运行都会拿 3072 PiB 去和真实剩余空间比较，**一个字节都没下就退出**。

**修法**：把常量改成 `RESERVE_GIB = 3.0`，与 help 文本（"in GiB"）和 `main()` 的换算一致。
默认值语义与 README 里 `--reserve 1.5` 的用法都恢复正常。

**为什么之前没人发现**：本会话之前 `ckpts/` 已经就位，所有文件都是 `[already present]`，
而这个 bug 恰好发生在**下载之前**的空间预检里，所以只要模型已存在就不会被触发。
是新增 `--variant asr` 时撞上的——新变体走同一段预检代码。

### 3.10 `audioread>=3.3` 无法满足（本次自己引入的 bug）

3.1 那次提交把 `audioread` 加进 core 依赖时写的是 `audioread>=3.3`。
但 **audioread 最新版就是 3.1.0**，3.3 根本不存在 → 约束不可满足，
**任何人全新安装 `pip install -e .` 都会直接失败**。本机之所以没暴露，
是因为当时是 `pip install audioread`（不带版本号）装的 3.1.0，绕过了自己写的约束。

**发现方式**：`pip install --dry-run -r requirements.txt` 与
`pip install --dry-run --ignore-installed`（模拟全新环境）报
`Could not find a version that satisfies the requirement audioread>=3.3`。

**修法**：改为 `audioread>=3.1`，两个 dry-run 复验均解析通过。

**教训**：写版本约束时**必须用 `pip install --dry-run` 验一遍可满足性**。
本机已安装的版本不会触发 pip 的解析检查，所以「本地能跑」完全不能证明「别人能装」。

### 3.11 ASR 权重迁入 `ckpts/`

**诉求**：把 `~/.cache/modelscope` 里的 SenseVoiceSmall 移进项目，便于整体迁移。

**改动**：`pe.py` 新增 `_default_asr_model()`，优先级为
`AUK_ASR_MODEL` 环境变量 → `ckpts/SenseVoiceSmall/model.pt` 存在则用本地目录 → 否则回落
`iic/SenseVoiceSmall`（保持原有自动下载行为）。路径用
`Path(__file__).resolve().parents[3]` 解析而**不用 CWD**——WebUI、CLI、`auk-gradio`
三者工作目录各不相同。

**可迁移性的实证方式**：不是「拷完看文件在不在」，而是

```bash
mv ~/.cache/modelscope ~/.cache/modelscope.bak      # 把原缓存整个移走
# → ASRCall.text = "Oh, excusecuse me Where did you get that…"  err=None
```

确认只靠 `ckpts/` 仍能转写后，才删除原缓存。

**`download_models.py` 新增 `--variant asr`，零新增代码路径**：
关键取证是 ModelScope 与 HF 分发的文件**字节完全相同**——用 HF 的 Git-LFS 指针 SHA-256
与本地副本比对，`model.pt`(`833ca2dc…`) 与 BPE model(`aa87f860…`) 均一致。
因此直接作为标准 HF variant 加进 `MODEL_SETS`，完整复用既有的
aria2c / LFS 校验 / manifest 合并 / 断点续传机制。

**只列 5 个文件**（funasr 实际加载的），并实测确认不含
`README.md`/`demo.py`/`example/`/`fig/` 也能加载。
ModelScope 版多带的 `tokens.json` **不需要**——HF 上根本没有该文件，
按 6 文件列表跑会 404，这也是一处需要实测才能发现的分歧。

**验证**：`--variant asr` 报 5 个文件全部 `[already present] sha256 ok`；
`--verify-only` 报 25/25 文件 OK（AuK 3 + Flash 3 + Qwen 14 + SenseVoiceSmall 5）。

### 3.12 文档更新



- `docs/WEBUI.md`：PE 章节重写（三种配法 + 新增「本地 LLM 服务」小节，含显存/eos/端口豁免等实测数据）；
  启动章节改为「五件事」；新增 `AUK_LOCAL_LLM`。
- `AGENTS.md`：新增 `scripts/local_llm_server.py` 条目、更新 `start_app.sh` 描述、
  **修正一条陈旧检查**（见 §7.6）。
- `README.md`：PE 章节加「No cloud LLM available?」段。
- `webui/app.py` 内 4 处「**没有本地兜底**」文案（页面 FAQ Q4、勾选即时提示、折叠面板说明、报错弹窗）
  已改为指向本地方案——文案与实际能力不符会误导用户。

---

## 4. 交付状态

### 4.1 commit 划分（`a5796ed..6d7f8f4`，均已推送）

刻意按「可独立回滚」拆分，而不是按文件堆：

```
7ed1673 fix(deps): cap gradio below 6.18                    ← 提交者原有的 gradio 上限
a556280 style(webui): rebrand header + indigo theme         ← 提交者原有的界面改动
01c8636 fix(deps): add audioread to core dependencies
132b21f feat(pe): serve the local Qwen2.5-Omni snapshot as an OpenAI-compatible endpoint
cd9d539 fix(pe): validate task classification and retry with the error appended
26f350e fix(pe): correct the inverted level target that silenced whisper-to-normal
cafff62 fix(webui): hand PE temp audio to the caller and honour .env for LLM config
c2681d0 docs: align the READMEs with the PE local-LLM and whisper-level behaviour
d70bd1a fix(download): correct the --reserve unit so the default no longer aborts
496255c feat(download): add --variant asr for the local SenseVoiceSmall weights
6e49e72 refactor(pe): resolve the local ASR weights from the repo, not the ModelScope cache
6d7f8f4 docs: document the in-repo SenseVoiceSmall location
```

`pe.py` 与 `pe.config.yaml` 同时出现在 `cd9d539`（分类重试）和 `26f350e`（耳语电平）里，
按 hunk 拆分，以便两个不相关的 bug 能分别回滚。拆分用了一个 hunk 过滤脚本做**非交互**暂存
（`git add -p` 是交互的，不适合自动化）。

### 4.2 工作区里不属于本次改动的内容

会话开始时工作区已有他人的未提交修改，已**单独成 commit** 以便需要时整体撤销：

- `a556280` 只含 `APP_TITLE`（腾讯开源 rebranding）+ CSS 配色
- `7ed1673` 只含 `gradio>=6.0.0,<6.18` 上限（`pyproject.toml` + README 对应段落）

`requirements.txt` 原为整个 conda 环境的 `pip freeze`（305 包，含 jupyter/conda/black/
`py-spy`），已重写为 20 条直接依赖并提交为 `e8521b7`。`.env` 始终被 `.gitignore` 忽略。

### 4.3 运行环境

```
本地 LLM 服务  127.0.0.1:8000   稳态 8514 MiB   /health → {"status":"ok","model":"qwen-omni-3b"}
WebUI          0.0.0.0:7860     HTTP 200       懒加载，AuK 首次生成时才载入
```
`start_app.sh` 自己拉起 LLM 服务时，日志写在 `outputs/local_llm_server.log`。
健康判据：日志行 `generated N tokens in Xs (eos=True, reserved=8.5 GiB)`
—— `eos=True` 且 `reserved` 稳定在 8.x 即正常（见 §7.9）。

## 5. 当前卡在哪 / 未解决

**没有阻塞项。** 但有几件已知的、有意未做的事：

### 5.1 本地 3B 模型的 PE 抽取精度不足（**已知限制，不是 bug**）

修好结构校验后仍存在质量问题：`把「but accepting」替换成「and living well」`
被抽出 `orig='but'`（丢了 `accepting`），而直引号版本能抽对。

**还有一个更严重的误分类**：中文指令 `用小声耳语的方式把这段话说出来。`
（`docs/COOKBOOK.md` 里明确的「normal → whisper」模板）被 3B 判成了 `whisper_edit / to_normal`
——**方向完全相反**，于是走了「转正常」的分支（见 §3.8）。
所以本地 3B 上，耳语双向转换可能整体不可用；要用请显式写英文模板
（`Convert this speech into a soft whisper…` / `Convert this whisper into normal speech…`），
或直接指回强模型。

- 重试只保证 **JSON 结构**合法，**不保证参数抽取/分类正确**。
- 根因是模型能力（3B），不是代码。
- **建议**：追求精度就把 `.env` 的 `LLM_BASE_URL` 指回 hy3 / deepseek（改三个值即可，代码零改动）。
  本地 3B 的定位是「零成本可用的兜底」。

### 5.2 显存偏紧

```
本地 LLM 稳态 ~8.5 GiB（生成中瞬时 ~15.7 GiB，见 §7.9）
  + AuK Base 推理峰值 ~10.9 GiB
  ≈ 19.4 / 23.5 GiB 稳态可行；瞬时峰值可能触及 ~23 GiB
```
多次实测能跑完（含用户真实界面操作），但余量不大，**别再加第三个模型**。
`AUK_LOCAL_LLM=0 ./start_app.sh 7860` 可关掉本地服务。


### 5.3 未做的验证

- **没做** GPU 以外的验证（无 CPU-only 路径测试）。
- **没做** 多用户并发压测（asyncio lock 串行化是按设计写的，未实测并发）。
- **没测** `whisper_edit / to_whisper` 子类型（只测了 `to_normal`），响度归一化走的是同一条
  `_normalize_audio_level` 分支，风险低但未实测。
- **没测** 云端 LLM（无 key）。`pe.py` 的重试改动对强模型是纯增益（不触发即无行为变化），
  但严格说未在云端后端上跑过。

### 5.4 后续可做（非阻塞）

- 若这批 commit 要走 PR 流程（上游历史里用的是 `Merge pull request`），
  可把 `a5796ed..6d7f8f4` 整理成 PR；PR 描述的素材 §8 已经齐了。
- `pe.config.yaml` 里还有一批同类物理常量（`duration.seconds_per_utf8_byte`、
  `vad.trim_padding_sec`、`loudness.*`）本次**只审了 `whisper.*` 四个**。
  详见 §10.6。

---

## 6. 如果要继续做，优先做什么

代码已全部落地并推送，下面按「值得做」排序，不是待办清单：

1. **给 PE 的分类/抽取加回归用例**。本仓库没有测试框架，可在 `scripts/` 加一个自检脚本
   （参考 `scripts/download_models.py` 的 CLI 风格），固定几条中文指令 + 期望
   `task_type`/`operation_subtype`。这样换模型或改 prompt 时能立刻发现退化——
   本次 3.8 那个静音 bug 和 3.6 那个 `operation_subtype: null` 都是靠人工对照才发现的。
2. **审 `pe.config.yaml` 里剩余的物理常量**（§10.6），并给每一类补上方向性不变量 + dBFS 验收。
3. **补测未覆盖路径**：`whisper_edit / to_whisper`、CPU offload 关闭、并发生成。
4. **上游走 PR** 的话整理 §4.1 那 12 个 commit 的描述。

## 7. 踩过的坑 —— 绝对不要再踩

### 7.1 别用 `nohup ... &` 起长期服务

会被启动它的 shell 一起回收。症状极具迷惑性：
服务刚 `/health` 通过，下一条命令调用就是 `APIConnectionError: Connection error.`，
而 `ps` 里进程已经没了、`log` 里只有 `Shutting down`。

**正确写法**：

```bash
setsid nohup python3 scripts/local_llm_server.py --port 8000 \
  > /tmp/opencode/llm.log 2>&1 < /dev/null & disown
```

（`bash` 工具有 120s 默认超时；后台起服务后**立刻**用短命令轮询健康端点，
不要把 `sleep 75` 之类和启动塞进同一条命令，否则整条命令被超时回收。）

### 7.2 别把「Qwen2.5-Omni-3B 已下载」当成「PE 可用」

它是 `AukInfer` 的 **text encoder**（`infer_auk.py` 里 `from_pretrained`），
负责把指令/参考音频编码成条件，**不是** OpenAI 兼容对话端点，PE 全程不碰它。
下载它 ≠ PE 能用。PE 的 LLM 是**完全独立**的需求。

### 7.3 依赖要用「实际调用链」验证，不要只看声明

本次三个缺包（`audioread`、`funasr`、`modelscope`）里有**两个在 `pyproject.toml` 里有声明**
（`gradio` extra），纯粹是环境没按 `pip install -e ".[gradio]"` 装全。
所以「文档说装了」≠「环境装了」。

**动手前先验**：

```bash
python3 -c "import audioread, funasr, modelscope; print('ok')"
pip list | grep -iE "^(audioread|funasr|modelscope|qwen-omni-utils) "
pip show qwen-omni-utils | grep -i requires
```

**装包后必须核对 torch 没被带偏**（funasr 依赖树很重）：

```bash
python3 -c "import torch,torchaudio;print(torch.__version__, torchaudio.__version__)"
```

### 7.4 读 `pe.py` 的失败信息要往上游追

PE 的报错都是 `PromptEnhancerError`，但真正的失败点在**别处**：
`_call_llm` 抛的会被包成「classify LLM 调用失败」，模型返回的**结构性**问题会一路冒到
`_normalize_params` 才炸，错误信息（`operation_subtype: None`）离病因很远。
`_extract_json` 的贪婪正则 `\{.*\}` 会让「答案 + 噪声」变成 JSON 解析失败，
所以**上游的停不下来**会以「JSON 坏了」的形式在下游报错。

### 7.5 编辑器/脚本里改 `start_app.sh` 时注意函数同名段落

本次 `free_port` 和 `free_gpu_memory` 里有一段**几乎逐字相同**的循环
（`is_self_tree "$pid" && …` / `info "终止占用…"`）。
`edit` 的 `oldString` 匹配到了**前者**，把 GPU 豁免逻辑插进了 `free_port`，
还把「终止占用**端口**的进程」文案覆盖成了「显存」。

**教训**：改这类脚本前先 `read` 完整函数；`oldString` 必须带足上下文唯一定位；
改完 `git diff` 逐行看。

### 7.6 `AGENTS.md` 里那条 Gradio 检查是陈旧的（已顺手修正）

原文档要求：

```bash
python3 -c "from auk.infer.infer_gradio import build_demo; assert build_demo()"
```

**在任何 checkout 里都必然抛** `RuntimeError: No valid model checkpoint was configured.`——
`infer_gradio.py:158` 的 `CKPT_PATHS: dict[str, str] = {}` 在模块级是**空 dict**，
只有 `main()` 的 argparse 会填。所以这不是「环境坏了」，是文档写错了。
已改成先 seed 再断言的形式，并用 `exit=0` 实测通过：

```bash
python3 -c "
import auk.infer.infer_gradio as g
g.CKPT_PATHS['AuK (Base)'] = 'ckpts/AuK/auk_base.safetensors'
g.CONFIG_PATHS['AuK (Base)'] = 'ckpts/AuK/config.yaml'
assert g.build_demo()
"
```

### 7.7 `ruff format` 会想重排既有代码 —— 别顺手动

`webui/app.py` 和 `pe.py` 在 HEAD 上 `ruff check` 就分别有 13 / 10 个既存告警，
`ruff format --check` 也已经是不通过状态。

**做法**：只保证**自己改的行**格式干净，用
`ruff format --diff <file> | grep '^@@'` 对比行号，确认待重排的区间**不在**你的改动里，
然后就不动。不要为了让 lint 全绿而重排整个文件（会制造巨大噪声 diff）。
本次实测：`webui/app.py` 我的改动在 654–820 行，`ruff format` 想改的 `@@ -1068` 在区间外。

### 7.8 改 `webui/app.py` 会影响 README 示例区

WebUI 的示例列表**在页面加载时解析 `README.md`**（`parse_readme_examples`）。
往 README 加 ```bash 代码块可能被误抓成示例。

判定规则在 `_parse_readme_command`：只有以 `auk-infer` 或 `_CMD_RE` 开头的行才会被当成示例，
所以以 `python scripts/...` 开头的块是安全的。**但要实测**：

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from webui.app import parse_readme_examples
ex = parse_readme_examples()
print('示例数:', sum(len(v) for v in ex.values()))
print('sections:', list(ex))
"
```
本次改动前后都是 5 条，未被污染。

### 7.9 显存核算要按「reserved」而不是「allocated」，并区分「稳态」与「瞬时峰值」

`torch.cuda.max_memory_allocated` / `memory_allocated` 看不出的坑：
PyTorch caching allocator 长期不把块还给驱动，`nvidia-smi` 显示的其实是 **reserved**。

本次实测三组数据（同一张卡、同一模型）：

| 状态 | LLM 服务占用 | 说明 |
|---|---|---|
| 无 `empty_cache()`（修复前） | **19.29 GiB** 且不回落 | 7.6 GiB 的模型把 AuK 挤到 OOM |
| 有 `empty_cache()`，请求之间 | **8.5 GiB 恒定** | 12 次采样（4s 间隔）完全平坦，无爬升 |
| 有 `empty_cache()`，生成进行中 | 瞬时 ~15.7 GiB | 生成结束后立刻回落到 8.5 GiB |

**所以看到 15–16 GiB 不代表泄漏**：那是 prefill + KV cache 的瞬时峰值，
`empty_cache()` 在 `_generate_sync` 的 `finally` 里，生成**过程中**还没执行。
判断是否泄漏要看**多次采样是否单调上升**，而不是单点读数。

交叉验证命令：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 配服务自己的日志行一起看：generated N tokens in Xs (eos=True, reserved=8.51 GiB)
for i in $(seq 1 12); do
  printf "%s  " "$(date +%H:%M:%S)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' '; echo
  sleep 4
done
```

`reserved=` 稳定在 8.x 且 `eos=True` 即为健康；`eos=False` 说明停不下来（见 §3.4 关键实现点第 1 条）。


### 7.10 跑 python 命令必须在仓库根目录

`ckpts/AuK/config.yaml` 里 `ckpts/Qwen2.5-Omni-3B` 是相对 **CWD** 解析的。
在别处运行会找不到 Qwen snapshot。`bash` 工具有 `workdir` 参数，优先用它而不是 `cd x && y`。

### 7.11 「没有报错」不等于「输出正常」——必须验音频本身

**这是本次最容易漏的一类 bug。** 耳语示例生成静音时，
PE 返回了 ✅、文件落盘了、时长正确、**没有异常、没有 NaN、peak 也非零**。
`infer_auk.py` 里现有的三道防线全部放过它：

```python
if gen_latent.shape[1] == 0:            raise RuntimeError(...)   # 长度
if torch.isnan(...).any() or isinf:     raise RuntimeError(...)   # NaN/Inf
if torch.isnan(gen_audio).any() ...      raise RuntimeError(...)   # NaN/Inf
```

**0.0006 的 RMS 不是 NaN，也不是 0。** 只验「成功 + 有文件 + 无 NaN」会漏掉整类电平错误。

**养成习惯：每次验收都打印 dBFS**，而且要和「关掉该功能」的基线对比：

```bash
python3 -c "
import torchaudio, math
w, sr = torchaudio.load('outputs/xxx.wav'); w = w.float()
rms = w.pow(2).mean().sqrt().item()
print(f'rms={rms:.6f}  {20*math.log10(max(rms,1e-12)):.1f} dBFS  peak={w.abs().max():.4f}  dur={w.shape[-1]/sr:.2f}s')
"
```
判读：正常语音约 **−26 dBFS**（rms 0.04–0.06）；低于 **−40 dBFS** 基本听不见；
**−63 dBFS** 就是静音。同一条指令**关掉 PE 再跑一次**做对照，才能确定病灶在功能开关的哪一侧。

**另一个通用教训**：定位时先做「**关掉可疑功能**」的对照实验。
本次正是靠「不勾选 PE 正常 / 勾选 PE 静音」这一组对照，把范围从模型侧（8 次生成、两个措辞、三个 seed）
压缩到 PE 的输出音频上，再往下追到具体配置项。
没有这组对照，很容易先去怀疑模型或种子。

### 7.12 配置里的物理量要有方向性不变量

`to_normal_target_rms < target_rms` 这种错误**没有任何运行期报错**——
归一化函数 happily地把波形乘上 gain 0.0143，写出一个合法 wav。
只有从「区间语义」层面检查才发现方向反了。

**做法**：
- 物理常量旁边写注释说明**为什么是这个量级**（-26 dBFS 对应正常语音电平）；
- 能在代码里结构化表达的不变量就别只靠注释——
  本次用 `max(WHISPER_TO_NORMAL_TARGET_RMS, WHISPER_TARGET_RMS)` 把「转正常必须比耳语响」
  变成代码保证，配置再被写错也不会静默产出静音。
- 看到 `0.000707945784384138` 这种 15 位有效数字的常量要**警惕**：
  它多半是某次实验的输出被直接粘进了配置，含义不明。查清它是怎么来的，别默认它对。


---

## 8. 验证命令与结果（提交 PR 时直接抄）

### 8.1 静态检查

```bash
ruff check scripts/local_llm_server.py     # All checks passed!
ruff format --check scripts/local_llm_server.py   # 1 file already formatted
ruff check src/auk/infer/pe.py              # Found 10 errors（与 HEAD 同数，无新增）
ruff check webui/app.py                     # Found 13 errors（与 HEAD 同数，无新增）
python3 -m compileall -q src scripts webui # 无输出
bash -n start_app.sh                        # 无输出
```

### 8.2 本地 LLM 服务

```bash
python scripts/local_llm_server.py --port 8000 &
curl -sS http://127.0.0.1:8000/health
# {"status":"ok","model":"qwen-omni-3b","vram_gib":7.64}
curl -sS http://127.0.0.1:8000/v1/models
# {"object":"list","data":[{"id":"qwen-omni-3b",...}]}
```
日志行 `generated N tokens in Xs (eos=True, reserved=8.51 GiB)` —— `eos=True` 且
`reserved` 稳定在 8.x 即为健康；`eos=False` 说明停不下来，`reserved` 飙到 19 GiB 说明 `empty_cache` 失效。

### 8.3 PE 端到端（真实 `run_task`，非仅 `PromptEnhancer.prepare`）

```bash
set -a; source ./.env; set +a
python3 <驱动 webui.app.run_task 的脚本>   # 参数见下
```

`run_task` 参数顺序（`webui/app.py`）：
```
variant, audio, instruction, gen_seconds, ref_text, gen_text,
nfe, cfg, seed, use_pe, llm_api_key, llm_base_url, llm_model
```
后三个传 `""` 才是真实页面默认行为（验证 env 回落）。

| 变体 | PE | 指令 | 结果 |
|---|---|---|---|
| AuK (Base) | ✗ | `Make it sound cheerful` | 28.00s，`outputs_20260927_051115.wav`，rms 0.2081 |
| AuK (Base) | ✓ | `把「but accepting」替换成「and living well」` | 60.4s，`outputs_20260927_054032.wav`，28.72s |
| AuK-Flash ⚡ | ✓ | 同上 | 53.7s，`outputs_20260927_054248.wav`，28.72s，**MD5 与 Base 不同** |
| AuK (Base) | ✓ | `Convert this whisper into normal speech…` | 50.4s，`outputs_20260927_055129.wav`，8.6s |
| AuK (Base) | ✓ | `把「but accepting」替换成「and living well」`（回归） | `outputs_20260927_055240.wav`，28.72s |
| AuK (Base) | ✓ | `Convert this whisper into normal speech…`（§3.8 修复后） | `outputs_20260927_061557.wav`，8.58s，**−26.7 dBFS** |

音频有效性（**必须看 dBFS，不能只看有无 NaN**，见 §7.11）：
```
shape (1, 689280) sr 24000 dur 28.72s rms 0.2089 nan False
shape (1, 205920) sr 24000 dur  8.58s rms 0.0465 (-26.7 dBFS)   ← 耳语示例，正常
```

### 8.4 启动脚本

以下为 2026-09-27 当天的真实输出样例（PID 已失效，仅供比对日志形态）：

```bash
setsid nohup ./start_app.sh 7860 > /tmp/opencode/final3.log 2>&1 < /dev/null & disown
```
关键日志行（全部实测出现）：
```
[INFO] 已加载 .env（检测到 LLM_API_KEY）。
[INFO] PE 端点  : http://127.0.0.1:8000/v1
[INFO] 端口 7860 已确认释放。
[INFO] 跳过本地 LLM 服务 PID 11488（端口 8000，WebUI 的 PE 依赖它）
[INFO] 终止占用显存的进程 PID 15351 ...
[INFO] 本地 LLM 已在 http://127.0.0.1:8000 运行，直接复用。
* Running on local URL:  http://0.0.0.0:7860
```
`grep -c "需要整数表达式"` → **0**。

显存（实测，稳态）：
```
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
11488, 8514 MiB      # 本地 LLM
15902, 1798 MiB      # WebUI（AuK 已懒加载时）
```

### 8.5 音频变体差异确认（别只看文件大小）

```bash
md5sum outputs/outputs_20260927_054032.wav outputs/outputs_20260927_054248.wav
# a2fd0d2e…  Base
# 5f6f77bb…  Flash   ← 不同，证明 Flash 确实走了 name=AuK-Flash 的蒸馏配方
```
两个文件**字节数相同**是正常的（时长都是 28.72s，字节数由采样点数决定），
所以判变体是否真的生效要看内容而非大小。

---

## 9. 关键文件与函数速查

| 位置 | 作用 | 本次是否改动 |
|---|---|---|
| `scripts/local_llm_server.py` | Qwen → OpenAI 兼容端点 | ✅ 新增 |
| `start_app.sh` | 启动器：`.env` / 清端口 / 清显存 / 起 LLM | ✅ 大改 |
| `webui/app.py` `run_task` | 主 generator；PE 临时音频清理的所有者 | ✅ |
| `webui/app.py` `_prepare_with_pe` | 调 PE；第 7 个返回值交出所有权 | ✅ |
| `webui/app.py` `llm_ready` | env 回落判定（**原本就正确**，被复用了） | — |
| `webui/app.py` `parse_readme_examples` | 解析 README 生成示例列表 | — |
| `src/auk/infer/pe.py` `_classify` / `_parse_classified` | 分类 + 校验 + 有界重试 | ✅ |
| `src/auk/infer/pe.py` `_prepare_audio` | 产出临时 wav 的地方；`to_normal` 的电平护栏在这里 | ✅ |
| `src/auk/infer/pe.py` `_normalize_audio_level` | 按 target_rms/lufs 归一化；**电平错误的现场** | — |
| `src/auk/infer/pe.py` `_extract_json` | 贪婪正则 `\{.*\}`（坑 7.4 的来源） | — |
| `src/auk/infer/infer_gradio.py` | **官方参考实现**，PE 生命周期处理正确 | — |
| `src/auk/infer/pe.config.yaml` | `api.llm.classify_max_attempts`、`runtime.whisper.to_normal_target_rms` | ✅ |
| `docs/WEBUI.md` | PE 章节 + 「本地 LLM 服务」 | ✅ |
| `AGENTS.md` | 陈旧检查修正 | ✅ |


---

## 10. 遗留风险（务必读完再改这块代码）

1. **3B 模型的 PE 分类/抽取精度有限**（§5.1）——结构合法 ≠ 参数正确，且已出现**方向性误分类**
   （把「转耳语」判成「转正常」）。生产建议指回强模型，或对耳语任务显式用英文模板。
2. **显存余量约 4 GiB**（§5.2）——本地 LLM + AuK 推理已验证可行，但别再加第三个模型。
3. **电平类缺陷没有任何运行期信号**（§7.11/§7.12）——`pe.config.yaml` 里还有大量物理常量
   （`duration.seconds_per_utf8_byte`、`f5.short_text_byte_threshold`、
   `vad.trim_padding_sec`、`loudness.*` 等），本次**只审了 `whisper.*` 四个**。
   新增或修改这类常量时，请一并做「开/关对照 + dBFS 验收」，否则同类静音问题还会再现。
4. **版本约束必须用 `pip install --dry-run` 验可满足性**（§3.10）——本机已装的版本不会触发
   pip 的解析检查，「本地能跑」完全不能证明「别人能装」。本次就自己踩了一次
   （`audioread>=3.3` 不可满足）。
5. **模型来源可能不一致**（§3.11）——ModelScope 与 HF 虽在本例中字节相同，但**文件列表不同**
   （ModelScope 多一个 `tokens.json`，HF 上没有）。换源时别假设两边等价，
   先按文件逐个验 SHA-256。
6. **`infer_gradio.py` 与 `webui/app.py` 是两套 PE 生命周期实现**——官方的
   `run_generate()` 与 `cleanup()` 在同一个 `try` 里是对的；`webui/app.py` 曾经把
   `cleanup()` 留在 prepare 函数中而生成在另一个函数里，导致临时音频被提前删除（§3.7）。
   **改 PE 的临时文件生命周期时，两处都要看。**
7. **`config.yaml` 里的 Qwen 路径是 CWD 相对的**（AGENTS.md 有提醒），
   而 `ckpts/SenseVoiceSmall` 是 `__file__` 相对的（§3.11）。两种解析方式并存，
   从非仓库根目录启动时注意区分。
