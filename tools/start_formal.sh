#!/bin/zsh
# Start (or resume) the formal runs of Step 5 from this folder, kept awake, with a log.
# Run from the root of the frozen step5 worktree:  zsh tools/start_formal.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p outputs/search/formal
caffeinate -dims /opt/miniconda3/envs/gphh/bin/python tools/run_formal.py "$@" 2>&1 | tee -a outputs/search/formal/launch.log
