#!/bin/bash

set -e

make

if [ "$1" == "-g" ]; then
    shift
    CMD="gdb --args ./mp"
else
    CMD=./mp
fi

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