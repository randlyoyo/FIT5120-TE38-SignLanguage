#!/bin/bash
# Wait on a PID, not on a name: matching a process by a string that also
# appears in the waiter's own command line is how the previous attempt
# deadlocked for 87 minutes.
while kill -0 65895 2>/dev/null; do sleep 10; done
echo "[drive] cache done: $(cat logs/mkcache3d_mtv.log | grep usable)"
export PYTHONUNBUFFERED=1
exec ./run_c.sh
