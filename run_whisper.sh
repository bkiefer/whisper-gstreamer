#!/bin/sh
LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH
python vadmicrowhisp.py "$@"
