#!/bin/bash
set +x

RAY_TMP_DIR="/ext_hdd2/jhna/mars/ray_tmp"
mkdir -p $RAY_TMP_DIR
export RAY_TMP_DIR
# Ray uses RAY_TMPDIR (no underscore) and/or `ray start --temp-dir`.
export RAY_TMPDIR="$RAY_TMP_DIR"
# Also steer Python/tempfile & some libs away from /tmp when possible.
export TMPDIR="$RAY_TMPDIR"

ray stop

CONFIG_PATH=$(basename $(dirname $0))

ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

ROLL_OUTPUT_DIR="/ext_hdd2/jhna/mars/runs/hanabi_selfplay/$(date +%Y%m%d-%H%M%S)"
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR
export ROLL_LOG_DIR=$ROLL_LOG_DIR
export ROLL_RENDER_DIR=$ROLL_RENDER_DIR
mkdir -p $ROLL_LOG_DIR $ROLL_RENDER_DIR

python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_hanabi_selfplay | tee $ROLL_LOG_DIR/custom_logs.log
