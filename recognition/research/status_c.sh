#!/bin/bash
# Arm C progress at a glance.  ./status_c.sh
cd "$(dirname "$0")"
# NOTE the bracket: pgrep -f metric_train would also match this script's own
# command line, which is exactly how the earlier run appeared to be alive
# while nothing was actually training.
P=$(pgrep -f "[m]etric_train.py")
if [ -n "$P" ]; then
  echo "● 训练中  pid $P  已运行 $(ps -o etime= -p $P | tr -d ' ')  内存 $(( $(ps -o rss= -p $P) / 1024 ))MB"
else
  echo "○ 训练进程不在"
fi
echo
echo "--- eval 记录 (每 5 轮) ---"
grep -E "^  ep " logs/train_c.log 2>/dev/null | tail -12 || echo "(还没到第一个 eval 点)"
echo
if [ -f runs/mtv3d/ckpt.pt ]; then
  venv/bin/python - <<'PY'
import torch
s = torch.load("runs/mtv3d/ckpt.pt", map_location="cpu", weights_only=False)
print(f"--- 断点 ---\n第 {s['epoch']}/120 轮   最佳 Valid AUC {s['best_auc']:.4f}")
PY
else
  echo "--- 断点 --- (第 5 轮后生成)"
fi
[ -f runs/mtv3d/result.json ] && echo && echo "✓ 已完成，结果见 runs/mtv3d/result.json"
exit 0
