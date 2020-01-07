#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

JSON=$1
KEYS=$2

$PYTHON tools/analyze_logs.py plot_curve $JSON --keys $KEYS --out demo/tmp/loss_curve.png