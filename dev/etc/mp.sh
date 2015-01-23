#!/bin/bash

set -e

function usage() {
################################################################################
    cat <<EOF
usage: ./run [-gsh] data_index

DESCRIPTION
   Compiles and executes mp.cu for the given data_index. data_index may be an
integer or 'all'.

OPTIONS
  -g     Run with debugger
  -s     Using only 1 OS thread
  -h     Show this message

EXAMPLES
  Run against dataset 0:
     ./run 0

  Run against dataset 1 using debugger and single-threaded:
     ./run -gs 0
EOF
    exit 1
}

CMD=./mp

while getopts "gsh" opt; do
    case $opt in
        g)
            if [ $(uname) == "Darwin" ]; then
                CMD="lldb ./mp"
            else
                CMD="gdb --args ./mp"
            fi
            ;;
        s)
            export EDU_CUDA_THREAD_COUNT=1
            ;;
        *)
            usage
            ;;
    esac
done

shift $((OPTIND-1))

if [ $# == 0 ] && [ -d data ]; then
    usage
fi

make

if [ -d data ]; then
    datasets=${1-all}

    if [ $datasets == all ]; then
        datasets='*'
    fi

    for x in data/$datasets; do
        echo "---"
        echo "--- TESTING DATASET $x"
        echo "---"

        $CMD $x
    done
else
    $CMD
fi