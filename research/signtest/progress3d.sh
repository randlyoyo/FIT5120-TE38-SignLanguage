#!/bin/sh
# 3D extraction progress, per split.
#   ./progress3d.sh          once
#   ./progress3d.sh -w       refresh every 60s
cd "$(dirname "$0")" || exit 1

show() {
  total=0
  printf "  %-10s %7s   %-22s %s\n" "split" "done" "" "target"
  for pair in Valid:6430 Train:38580 Test_STU:6430 Test_ITW:6430 \
              Test_TED:6430 Test_SYN:6430 Test_MTV:12860; do
    s=${pair%:*}; want=${pair#*:}
    n=$(ls "keypoints3d/$s" 2>/dev/null | wc -l | tr -d ' ')
    total=$((total + n))
    pct=$((n * 100 / want))
    filled=$((pct / 5))
    bar=""
    i=0
    while [ $i -lt $filled ]; do bar="$bar#"; i=$((i + 1)); done
    printf "  %-10s %7s   [%-20s] %3s%%  /%s\n" "$s" "$n" "$bar" "$pct" "$want"
  done
  echo "  ------------------------------------------------------------"
  printf "  %-10s %7s   %19s %3s%%  /83590\n" "TOTAL" "$total" "" \
    "$((total * 100 / 83590))"
  echo
  # More than one running job means two are racing over the same clips, which
  # is what dropped throughput 5x once already. Zero means it needs restarting.
  jobs=$(pgrep -f run_3d.sh | wc -l | tr -d ' ')
  fails=$(grep -ac FAIL logs/extract3d.log 2>/dev/null)
  [ -z "$fails" ] && fails=0
  printf "  jobs %s  (1 = healthy)   failures %s   output %s\n" \
    "$jobs" "$fails" "$(du -sh keypoints3d 2>/dev/null | cut -f1)"
}

if [ "$1" = "-w" ]; then
  while true; do clear; date; echo; show; sleep 60; done
else
  show
fi
