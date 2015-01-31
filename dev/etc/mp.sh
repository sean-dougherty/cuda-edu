#!/bin/bash

set -e

function usage() {
################################################################################
    cat <<EOF
usage: ./run [-gpsh] data_index

DESCRIPTION
   Compiles and executes mp.cu for the given data_index. data_index may be an
integer or 'all'.

OPTIONS
  -g     Run with debugger
  -p     Disable debugger plugin
  -s     Using only 1 OS thread
  -h     Show this message

EXAMPLES
  Run against dataset 0:
     ./run 0

  Run against dataset 1 using debugger:
     ./run -g 1

  Run against dataset 1 using debugger without plugin:
     ./run -gp 1

EOF
    exit 1
}

################################################################################
#
# Locate install dir
#
################################################################################
EDUDIR=$PWD
while [ ! -d $EDUDIR/dev/db ] || [ ! -d $EDUDIR/dev/runtime ]; do
    EDUDIR=$(dirname $EDUDIR)
    
    if [ -z "$EDUDIR" ] || [ "$EDUDIR" == "/" ]; then
        echo "Failed determining cuda-edu install dir!" >&2
        exit 1
    fi
done

################################################################################
#
# Process options
#
################################################################################
opt_debugger=false
opt_single_threaded=false
opt_disable_plugin=false

while getopts "gsph" opt; do
    case $opt in
        g)
            opt_debugger=true
            ;;
        s)
            opt_single_threaded=true
            ;;
        p)
            opt_disable_plugin=true
            ;;
        *)
            usage
            ;;
    esac
done

single_threaded=false

if $opt_debugger; then
    if [ $(uname) == "Darwin" ]; then
        CMD="lldb ./mp"
    else
        if $opt_disable_plugin; then
            CMD="gdb --args ./mp"
        else
            CMD="gdb -ex 'source $EDUDIR/dev/db/gdb-plugin.py' --args ./mp"
            single_threaded=true
        fi
    fi
else
    CMD=./mp
fi

if $opt_single_threaded; then
    single_threaded=true
fi

if $single_threaded; then
    export EDU_CUDA_THREAD_COUNT=1
fi

shift $((OPTIND-1))

if [ $# == 0 ] && [ -d data ]; then
    usage
fi

################################################################################
#
# Build
#
################################################################################
make

################################################################################
#
# Run
#
################################################################################
if [ -d data ]; then
    datasets=${1-all}

    if [ $datasets == all ]; then
        datasets='*'
    fi

    for x in data/$datasets; do
        echo "---"
        echo "--- TESTING DATASET $x"
        echo "---"

        eval $CMD $x
    done
else
    eval $CMD
fi