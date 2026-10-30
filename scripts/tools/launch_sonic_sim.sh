#!/usr/bin/env bash
# Launch SONIC sim + deploy in a tmux session (2 panes, horizontal split).
#
# Layout:
#   +-----------------+-----------------+
#   | sim             | deploy          |  ** press ] then Enter in 'deploy' **
#   +-----------------+-----------------+
#
# Companion to pico_viz. Run pico_viz first (body retarget publishes pose
# on tcp:5556); this script spins up MuJoCo sim and the SONIC ZMQ deploy
# that subscribes to that pose stream.
#
# Usage:
#   bash scripts/tools/launch_sonic_sim.sh
#
# Navigation:
#   Ctrl-b then arrow keys     switch pane
#   Ctrl-b z                   zoom pane fullscreen
#   Ctrl-b d                   detach (keeps running)
#   tmux attach -t sonic_sim   re-attach
#   tmux kill-session -t sonic_sim    stop both

SESSION="sonic_sim"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GR00T_ROOT="$REPO_ROOT/thirdparty/GR00T-WholeBodyControl"

tmux kill-session -t "$SESSION" 2>/dev/null

# Pane 0 — MuJoCo sim (separate venv)
SIM=$(tmux new-session -d -s "$SESSION" -x 220 -y 50 -c "$GR00T_ROOT" -P -F '#{pane_id}')
tmux send-keys -t "$SIM" "source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py" Enter
sleep 3  # let sim spin up before deploy connects

# Pane 1 — SONIC deploy (needs ] then Enter interactively)
DEPLOY=$(tmux split-window -h -t "$SIM" -c "$GR00T_ROOT/gear_sonic_deploy" -P -F '#{pane_id}')
tmux send-keys -t "$DEPLOY" "bash deploy.sh sim --input-type zmq --zmq-host localhost" Enter

tmux select-layout -t "$SESSION" even-horizontal
tmux select-pane -t "$DEPLOY"
tmux display-message -t "$SESSION" "Focus 'deploy' pane: press ] then Enter to start policy"
tmux attach-session -t "$SESSION"
