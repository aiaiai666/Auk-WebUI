#!/usr/bin/env bash
# =============================================================================
#  AuK WebUI 一键启动脚本
#
#  用法:
#      ./start_app.sh              # 默认端口 7860
#      ./start_app.sh 7860         # 显式指定端口
#      AUK_PRELOAD=1 ./start_app.sh   # 启动即加载模型（放弃懒加载）
#
#  启动前会自动做三件事:
#      1. 确保 fuser 可用（缺则装 psmisc）
#      2. 无交互终止占用该端口的旧进程，并轮询确认端口真正释放
#      3. 清理占用显存的其它进程，保证本次运行独占 GPU
#
#  注意: 脚本会"杀掉占用目标端口的进程"和"杀掉其它 GPU 计算进程"，
#        但**不会**动与本次运行无关的其它进程（jupyter / sshd 等都不会受影响）。
# =============================================================================

set -Eeuo pipefail

# ------------------------------ 可配置参数 -----------------------------------
PORT="${1:-7860}"
HOST="0.0.0.0"                       # 0.0.0.0 才能让外部设备访问
PORT_WAIT_SECONDS=3                  # 端口释放轮询上限
PORT_POLL_INTERVAL=0.1
GPU_WAIT_SECONDS=3
PYTHON_BIN="${PYTHON_BIN:-}"
[ -x /opt/conda/bin/python ] && PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
# HuggingFace 镜像。
# 注意：本镜像环境会在 PID 1 里预设一个不可达的 HF_ENDPOINT（构建期残留），
# 若沿用会让运行时的 HF 访问全部失败。这里统一改用可用的加速镜像，
# 需要换镜像时通过 AUK_HF_ENDPOINT 指定。
export HF_ENDPOINT="${AUK_HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False

# ------------------------------ 目录定位 -------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$SCRIPT_DIR"                     # 保证 config.yaml 里 ckpts/ 相对路径可解析

# ------------------------------ 日志工具 -------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $(date '+%F %T') $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%F %T') $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $(date '+%F %T') $*" >&2; }
die()  { err "$*"; exit 1; }

# ---------------------- 工具: 判断 PID 是否属于自身进程树 ---------------------
is_self_tree() {
    local target="$1" cur="$$"
    while [ -n "$cur" ] && [ "$cur" -gt 1 ]; do
        [ "$cur" = "$target" ] && return 0
        cur="$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ' || true)"
    done
    [ "$target" = "1" ] && return 0     # 绝不动 init/supervisord
    return 1
}

# ---------------------- 步骤 1: 确保 fuser 可用 -------------------------------
ensure_fuser() {
    if command -v fuser >/dev/null 2>&1; then
        return 0
    fi
    warn "未找到 fuser，尝试安装 psmisc ..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get install -y psmisc >/dev/null 2>&1 || true
    fi
    command -v fuser >/dev/null 2>&1 && { info "psmisc 安装完成"; return 0; }
    die "fuser 不可用且无法自动安装 psmisc；请手动执行: apt-get install -y psmisc"
}

# ---------------------- 步骤 2: 释放端口 -------------------------------------
# 轮询直到端口真正释放（最多 PORT_WAIT_SECONDS 秒）
wait_port_released() {
    local waited=0
    while fuser "${PORT}/tcp" >/dev/null 2>&1; do
        if [ "$waited" -ge "$PORT_WAIT_SECONDS" ]; then
            return 1
        fi
        sleep "$PORT_POLL_INTERVAL"
        waited="$(awk -v a="$PORT_POLL_INTERVAL" -v b="$waited" 'BEGIN{printf "%.1f", a+b}')"
    done
    return 0
}

free_port() {
    if ! fuser "${PORT}/tcp" >/dev/null 2>&1; then
        info "端口 ${PORT} 空闲，无需清理。"
        return 0
    fi

    local pids
    pids="$(fuser "${PORT}/tcp" 2>/dev/null || true)"
    warn "端口 ${PORT} 被占用，占用进程: ${pids//$'\n'/ }"

    # 只终止这些 PID，且跳过自身进程树，绝不误伤其它进程
    local pid killed=0
    for pid in $pids; do
        is_self_tree "$pid" && { info "跳过自身进程树 PID $pid"; continue; }
        info "终止占用端口的进程 PID $pid ..."
        kill -TERM "$pid" 2>/dev/null || true
        killed=1
    done

    if [ "$killed" = "1" ] && ! wait_port_released; then
        warn "端口 ${PORT} 在 ${PORT_WAIT_SECONDS}s 内未释放，升级为 SIGKILL ..."
        for pid in $(fuser "${PORT}/tcp" 2>/dev/null || true); do
            is_self_tree "$pid" && continue
            kill -KILL "$pid" 2>/dev/null || true
        done
        wait_port_released || die "端口 ${PORT} 仍被占用，已放弃启动。请人工检查。"
    fi

    fuser "${PORT}/tcp" >/dev/null 2>&1 && die "端口 ${PORT} 仍被占用，已放弃启动。"
    info "端口 ${PORT} 已确认释放。"
}

# ---------------------- 步骤 3: 清理显存占用 ---------------------------------
free_gpu_memory() {
    command -v nvidia-smi >/dev/null 2>&1 || { info "无 nvidia-smi，跳过显存清理。"; return 0; }

    local pids pid killed=0
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"

    if [ -z "${pids// /}" ]; then
        info "GPU 无计算进程，显存已空闲。"
        return 0
    fi

    warn "检测到占用显存的进程: ${pids//$'\n'/ }，正在清理以保证独占 ..."
    for pid in $pids; do
        is_self_tree "$pid" && { info "跳过自身进程树 PID $pid"; continue; }
        info "终止 GPU 进程 PID $pid ..."
        kill -TERM "$pid" 2>/dev/null || true
        killed=1
    done

    if [ "$killed" = "1" ]; then
        local waited=0
        while [ "$waited" -lt "$GPU_WAIT_SECONDS" ]; do
            pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
            [ -z "${pids// /}" ] && break
            sleep "$PORT_POLL_INTERVAL"
            waited="$(awk -v a="$PORT_POLL_INTERVAL" -v b="$waited" 'BEGIN{printf "%.1f", a+b}')"
        done
        # 仍在占用的补一刀 SIGKILL
        pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
        for pid in $pids; do
            is_self_tree "$pid" && continue
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi

    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
    if [ -n "${pids// /}" ]; then
        warn "仍有进程占用显存: ${pids//$'\n'/ }（可能属其它用户，继续启动）"
    else
        info "显存已清理干净，本次运行独占 GPU。"
    fi
}

# ---------------------- 主流程 ------------------------------------------------
main() {
    info "=========================================================="
    info " AuK WebUI 启动中 ..."
    info " 项目目录 : ${SCRIPT_DIR}"
    info " Python  : ${PYTHON_BIN}  ($(${PYTHON_BIN} --version 2>&1))"
    info " 监听地址 : http://${HOST}:${PORT}  (0.0.0.0 允许外部设备访问)"
    info " HF 镜像  : ${HF_ENDPOINT}"
    info "=========================================================="

    [ -f "webui/app.py" ] || die "未找到 webui/app.py，请在本项目根目录运行。"
    [ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python3)" || die "找不到 python3"

    mkdir -p outputs

    ensure_fuser
    free_port
    free_gpu_memory

    local extra_args=()
    [ "${AUK_PRELOAD:-0}" = "1" ] && extra_args+=(--preload)

    info "启动 WebUI 主程序 ..."
    exec "$PYTHON_BIN" -m webui.app --host "$HOST" --port "$PORT" "${extra_args[@]+"${extra_args[@]}"}"
}

main "$@"
