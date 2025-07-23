#!/bin/sh
#set -x
scrdir=`dirname $0`

if test -z "$ops_dir"; then
    if test -z "$pref"; then
        if test -n "$CONDA_PREFIX"; then
            pref="$CONDA_PREFIX"
        else
            pref="/usr"
        fi
    fi
    ops_dir=$(find "$pref" -name libcudnn_ops\*)
    if test -z "$ops_dir"; then
        echo "libcudnn_ops not found, you may have to set it in this script"
    else
        ops_dir=$(dirname "$ops_dir")
    fi
fi
LD_LIBRARY_PATH="$ops_dir"
export LD_LIBRARY_PATH
python3 $scrdir/vadmicrowhisp.py "$@"
