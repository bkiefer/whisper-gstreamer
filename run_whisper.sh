#!/bin/sh
scrdir=`dirname $0`

if test -z "$ops_dir"; then
    ops_dir=$(dirname $(find $CONDA_PREFIX -name libcudnn_ops_infer.so\*))
    if test -z "$ops_dir"; then
        echo "libcudnn_ops_infer not found, you may have to set it in this script"
    fi
fi
LD_LIBRARY_PATH="$ops_dir"
export LD_LIBRARY_PATH
python $scrdir/vadmicrowhisp.py "$@"
