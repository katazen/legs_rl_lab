#!/usr/bin/env bash
# 复用实机启动链路，仅跳过 RL 的准备姿态与到位补偿；保留 P/A 开跑确认。
set -e
H1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$H1/start_real.sh" --start-from-current "$@"
